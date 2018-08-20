import os
import sys
import copy
import json
import time
import nltk
import numpy
import string
import pickle
import logging
import itertools
import threading
import subprocess
import collections
from statistics import mean
from datetime import datetime
from stop_words import get_stop_words
from sklearn import svm, model_selection
from collections import defaultdict, OrderedDict
from gensim import models, similarities, corpora

MANNAME = 'manual'
SESSIONS = 'sessions'
LSIMODEL = 'lsimodel'
ALIASPATH = 'alias.txt'
LOGNAME = 'log_dataset'
RAWPATH = 'raw_dataset'
LOGPATH = '/v/global/appl/appmw/tam-ar-etl/data/shellmask_dev/shelllogreview/logs'
PERIODS = ['201707', '201708', '201709', '201710', '201711', '201712', '201801', '201802', '201803', '201804', '201805',
           '201806', '201807', '201808']


class Activity:
    def __init__(self, activity: dict):
        self.command = activity['command']
        self.seqno = activity['sequence_number']
        self.datetime = activity['datetime']
        self.id =activity.get('id')


class Session:
    def __init__(self, session: dict):
        self.activities = {}
        if 'activities' in session:
            for activity in session['activities']:
                self.activities[activity['sequence_number']] = Activity(activity)
        self.uuid = session['uuid']
        self.starttime = session.get('starttime')
        self.endtime = session.get('endtime')
        self.hostname = session.get('hostname')
        self.rule = session.get('rule')
        self.runas = session.get('runas')
        self.subject = session.get('subject')
        self.tags = session.get('tags', [])
        self.ticketas = session.get('ticketas')
        self.tooltype = session.get('toolType')
        self.userid = session.get('userid')

    def update(self, another):
        self.activities.update(another.activities)
        if self.starttime is None:
            self.starttime = another.starttime
        if self.endtime is None:
            self.endtime = another.endtime
        if self.hostname is None:
            self.hostname = another.hostname
        if self.rule is None:
            self.rule = another.rule
        if self.runas is None:
            self.runas = another.runas
        if self.subject is None:
            self.subject = another.subject
        for tag in another.tags:
            self.tags.append(tag)
        if self.ticketas is None:
            self.ticketas = another.ticketas
        if self.tooltype is None:
            self.tooltype = another.tooltype
        if self.userid is None:
            self.userid = another.userid
        return self


class Shell_Command:
    def __init__(self, data: dict):
        self.raw = data['n']
        self.description = data['d']


def load_shell_commands(dumped: bool = True) -> {str: str}:
    if dumped:
        with open(MANNAME + '.pickle', 'rb') as man_file:
            manpages = pickle.load(man_file)
    else:
        s = subprocess.Popen('bash', stdout=subprocess.PIPE, stdin=subprocess.PIPE, shell=True)
        s.stdin.write(b'compgen -c\n')
        s.stdin.close()
        manpages = {}
        commands = s.stdout.read().decode('utf-8').split()
        for command in commands:
            if sys.version_info[1] > 4:
                man = subprocess.run(['man', command], stdout=subprocess.PIPE).stdout.decode('utf-8')
                if man:
                    manpages[command] = man
            else:
                try:
                    man = subprocess.check_output(['man', command]).decode('utf-8')
                    manpages[command] = man
                except subprocess.CalledProcessError:
                    pass
        man_file = open(MANNAME + '.pickle', 'wb')
        pickle.dump(manpages, man_file)
        man_file.close()
    logging.info('load {} known shell commands in warehouse with manual'.format(len(manpages.keys())))
    return manpages


def shell_commands_embedding(manpages: {str: str}, dumped: bool = True, similar_commands: [str] = []) -> {str: [float]}:
    if dumped:
        with open(LSIMODEL + '.pickle', 'rb') as lsi_file:
            return pickle.load(lsi_file)

    stop_words = get_stop_words('en')
    stemmer = nltk.stem.SnowballStemmer('english')
    commands = []
    for command, manual in manpages.items():
        manpages[command] = []
        commands.append(command)
        for word in nltk.tokenize.word_tokenize(manual.lower()):
            if word not in stop_words and word not in string.punctuation:
                # TODO: can remove the word which only appears once
                manpages[command].append(stemmer.stem(word))
    dictionary = corpora.Dictionary(manpages.values())
    corpus = [dictionary.doc2bow(text) for text in manpages.values()]
    tfidf = models.TfidfModel(corpus)
    lsi = models.LsiModel(tfidf[corpus], id2word=dictionary, num_topics=100)
    index = similarities.MatrixSimilarity(lsi[corpus])

    lsimodel = {}
    for command in manpages.keys():
        lsimodel[command] = lsi[corpus[commands.index(command)]]

    def find_similar_commands(command: str):
        threshold = 0.8
        if command in commands:
            sort_sims = sorted(enumerate(index[lsimodel[command]]), key=lambda item: -item[1])
            print('following commands are similar to \'' + '\033[1m' + '{}'.format(command) + '\033[0m' + '\':')
            for sim in sort_sims:
                if sim[1] > threshold:
                    print('\t\033[1m' + '{}'.format(commands[sim[0]]) + '\033[0m' + ' ({})'.format(sim[1]))

    for similar_command in similar_commands:
        find_similar_commands(similar_command)

    lsi_file = open(LSIMODEL + '.pickle', 'wb')
    pickle.dump(lsimodel, lsi_file)
    lsi_file.close()

    return lsimodel


def load_dataset(manpages: {str: str}, dumped: bool = True, multithread: bool = False, summary_top: int = 8):
    if dumped:
        with open(SESSIONS + '.pickle', 'rb') as sessions_file:
            return pickle.load(sessions_file)

    def load_alias() -> {str: str}:
        aliases = {}
        try:
            with open(ALIASPATH, 'r') as alias_file:
                for line in alias_file:
                    aliases[line.split('=', 1)[0].strip('\'')] = line.split('=', 1)[1].strip('\'').strip('\n')
        except FileNotFoundError:
            print('alias file not found, please run the command to generate: \n\talias > ' + ALIASPATH)
        return aliases

    def read_dataset(period: str, subperiod: str, index: str):
        if not os.path.exists(RAWPATH):
            os.mkdir(RAWPATH)
        logging.debug('reading dataset from json in following period: {}'.format(period + '_' + subperiod + '_' + index))
        sessions = {}
        for root, dirs, files in os.walk(LOGPATH + '/' + period + '/' + subperiod + '/' + index):
            for name in files:
                file_dir = os.path.join(root, name)
                logging.info('current read json file: {}'.format(file_dir.split('logs/')[1]))
                with open(file_dir, encoding='utf-8') as raw_log_file:
                    data = json.load(raw_log_file)
                session = Session(data)
                if session.uuid in sessions:
                    sessions[session.uuid].update(session)
                else:
                    sessions[session.uuid] = session
            log_file = open(RAWPATH + '/' + LOGNAME + '_' + period + '_' + subperiod + '_' + index + '.pickle', 'wb')
            pickle.dump(sessions, log_file)
            log_file.close()
        logging.debug('the size of log is {} MB'.format(sys.getsizeof(sessions) / 1000 / 1000))

    if multithread:
        pool = []
        for period in PERIODS:
            for subperiod in os.listdir(LOGPATH + '/' + period):
                if subperiod == '201803270801' or subperiod == '201804230547':
                    continue
                for index in os.listdir(LOGPATH + '/' + period + '/' + subperiod):
                    pool.append(threading.Thread(target=read_dataset, args=(period, subperiod, index, )))
        for thread in pool:
            thread.start()
        for thread in pool:
            thread.join()

    aliases = load_alias()
    commands = defaultdict(int)
    days = defaultdict(int)
    weekday_time = defaultdict(int)
    month_time = defaultdict(int)
    latest_date = datetime.now()

    all_sessions = {}

    for raw_file in os.listdir(RAWPATH):
        logging.info('load dataset from pickle in following file: {}'.format(raw_file))
        with open(RAWPATH + '/' + raw_file, 'rb') as log_file:
            sessions = pickle.load(log_file)
        for session in sessions.values():
            for activity in list(session.activities.values()):
                if activity.command is not None:
                    activity.command = activity.command.strip()
                    if len(activity.command.split()):
                        for alias in aliases.keys():
                            if activity.command.startswith(alias + ' '):
                                activity.command = activity.command.replace(alias, aliases[alias], 1)
                        commands[activity.command.split()[0]] += 1
                        activity_time = datetime.fromtimestamp(time.mktime(time.localtime(int(activity.datetime))))
                        latest_date = min(latest_date, activity_time)
                        days[activity_time.strftime('%Y-%m-%d')] += 1
                        weekday_time[activity_time.weekday()] += 1
                        month_time[activity_time.month] += 1
                    else:
                        session.activities.pop(activity.seqno)
                else:
                    session.activities.pop(activity.seqno)
            if session.uuid in all_sessions:
                all_sessions[session.uuid].update(session)
            else:
                all_sessions[session.uuid] = session

    logging.info(
        'current load {0} with {1} unique commands from pickle'.format(sum(commands.values()), len(commands.keys())))
    known_commands = {}
    for command in set(commands.keys()) & set(manpages.keys()):
        known_commands[command] = commands[command]
    logging.info(
        'already known {0} with {1} unique commands.'.format(sum(known_commands.values()), len(known_commands.keys())))

    sessions_file = open(SESSIONS + '.pickle', 'wb')
    pickle.dump(all_sessions, sessions_file)
    sessions_file.close()

    basic_info = (sum(commands.values()), sum(known_commands.values()), len(known_commands.keys()), len(days.keys()))
    summary_info = (weekday_time, month_time, dict(collections.Counter(known_commands).most_common(summary_top)))
    other_info = (latest_date.strftime('%b %d, %Y'), )

    return all_sessions, basic_info, summary_info, other_info


def outlier_detect(sessions: {str: Session}, lsimodel: {str: [(int, float)]}, window: int = 10, test_size: float = 0.5,
                   folds: int = 1, topno: int = 10):
    dataset = defaultdict(list)
    for session in sessions.values():
        if session.userid is not None:
            for seqno, activity in sorted(session.activities.items(), key=lambda item: item[0]):
                if activity.command.split()[0] in lsimodel.keys():
                    dataset[session.userid].append(activity.command.split()[0])

    for userid, commands in dataset.items():
        dataset[userid] = list(map(list, zip(*[iter(commands)] * window)))
    dataset = OrderedDict(sorted(dataset.items(), key=lambda item: len(item[1]), reverse=True)[:topno])
    commandset = copy.deepcopy(dataset)

    for userid, commands_windows in dataset.items():
        for i, commands_window in enumerate(commands_windows):
            data = []
            for command in commands_window:
                data.append([item[1] for item in lsimodel[command]])
            dataset[userid][i] = list(map(mean, zip(*data)))

    for userid in dataset.keys():
        for times in range(0, folds):
            logging.debug('userid {0} has {1} commands'.format(userid, len(dataset[userid])))
            train_dataset, test_dataset, train_commandset, test_commandset = model_selection.train_test_split(
                dataset[userid], commandset[userid], test_size=test_size)

            clf = svm.OneClassSVM(nu=0.05, kernel="poly")
            clf.fit(train_dataset)
            train_pred = clf.predict(train_dataset)
            test_pred = clf.predict(test_dataset)
            n_malicious_train = train_pred[train_pred == -1].size
            n_malicious_test = test_pred[test_pred == -1].size
            logging.info(
                'user {0} malicious no of trainset is {1} with all {2} in fold {3}'.format(userid, n_malicious_train,
                                                                                           len(train_pred), times))
            logging.info(
                'user {0} malicious no of testset is {1} with all {2} in fold {3}'.format(userid, n_malicious_test,
                                                                                          len(test_pred), times))

            if (n_malicious_train / len(train_pred)) < 0.1:
                train_commandset = numpy.array(train_commandset)
                test_commandset = numpy.array(test_commandset)
                malicious_train = set(itertools.chain(*train_commandset[train_pred == -1]))
                logging.info('user {0} malicious command {1} in trainset'.format(userid, malicious_train))
                malicious_test = set(itertools.chain(*test_commandset[test_pred == -1]))
                logging.info('user {0} malicious command {1} in testset'.format(userid, malicious_test))


def main():
    logging.basicConfig(level=logging.DEBUG)

    manpages = load_shell_commands()
    # lsimodel = shell_commands_embedding(manpages)
    sessions, basic_info, summary_info, other_info = load_dataset(manpages, dumped=False, multithread=True)
    # outlier_detect(sessions, lsimodel, window=1)


if __name__ == '__main__':
    main()
