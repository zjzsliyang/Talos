import os
import sys
import json
import nltk
import string
import pickle
import logging
import threading
import subprocess
from collections import defaultdict
from stop_words import get_stop_words
from sklearn import svm, model_selection
from gensim import models, similarities, corpora


MANNAME = 'manual'
LSIMODEL = 'lsi.model'
ALIASPATH = 'alias.txt'
LOGNAME = 'log_dataset'
LOGPATH = '/v/global/appl/appmw/tam-ar-etl/data/shellmask_dev/shelllogreview/logs'
SUBNAMES = ['0, ''1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F']
PERIODS = ['201707', '201708', '201709', '201710', '201711', '201712', '201801', '201802', '201803', '201804', '201805',
           '201806', '201807']


class Activity:
    def __init__(self, activity: dict):
        self.command = activity['command']
        self.datetime = activity['datetime']
        self.id = activity['id']
        self.seqno = activity['sequence_number']


class Session:
    def __init__(self, session: dict):
        self.activities = {}
        if 'activities' in session:
            for activity in session['activities']:
                self.activities[activity['sequence_number']] = Activity(activity)
        self.starttime = session['starttime']
        self.endtime = session['endtime']
        self.hostname = session['hostname']
        self.rule = session['rule']
        self.runas = session['runas']
        self.subject = session['subject']
        self.tags = session['tags']
        self.ticketas = session['ticketas']
        self.tooltype = session['toolType']
        self.userid = session['userid']
        self.uuid = session['uuid']

    def merge(self, session: dict):
        another = Session(session)
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


def shell_commands_embedding(manpages: {str: str}, similar_commands: [str] = []) -> {str: [float]}:
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

    return lsimodel


def load_dataset(manpages: {str: str}, multithread: bool = False) -> [Session]:
    def load_alias() -> {str: str}:
        aliases = {}
        try:
            with open(ALIASPATH, 'r') as alias_file:
                for line in alias_file:
                    aliases[line.split('=', 1)[0].strip('\'')] = line.split('=', 1)[1].strip('\'').strip('\n')
        except FileNotFoundError:
            print('alias file not found, please run the command to generate: \n\talias > ' + ALIASPATH)
        return aliases

    def read_dataset(period: str, subname: str):
        logging.info('reading dataset from json in following period: {}'.format(period))
        sessions = {}
        for root, dirs, files in os.walk(LOGPATH + '/' + period + '/' + subname):
            for name in files:
                file_dir = os.path.join(root, name)
                logging.debug('current read json file: {}'.format(file_dir.split('logs/')[1]))
                with open(file_dir, encoding='utf-8') as raw_log_file:
                    data = json.load(raw_log_file)
                session = Session(data)
                if session.uuid in sessions:
                    sessions[session.uuid].merge(session)
                else:
                    sessions[session.uuid] = session
            log_file = open(LOGNAME + '_' + period + '_' + subname + '.pickle', 'wb')
            pickle.dump(sessions, log_file)
            log_file.close()
        logging.info('the size of log is {} MB'.format(sys.getsizeof(sessions) / 1000 / 1000))

    if multithread:
        pool = []
        for period in PERIODS:
            for subname in SUBNAMES:
                pool.append(threading.Thread(target=read_dataset, args=(period, subname)))
        for thread in pool:
            thread.start()
        for thread in pool:
            thread.join()

    aliases = load_alias()
    commands = defaultdict(int)
    for period in PERIODS:
        logging.info('load dataset from pickle in following period: {}'.format(period))
        for subname in SUBNAMES:
            with open(LOGNAME + '_' + period + '_' + subname + '.pickle', 'rb') as log_file:
                sessions = pickle.load(log_file)
            for session in sessions.values():
                for activity in session.activities:
                    if activity.command is not None:
                        activity.command = activity.command.strip()
                        if len(activity.command.split()):
                            for alias in aliases.keys():
                                if activity.command.startswith(alias + ' '):
                                    activity.command = activity.command.replace(alias, aliases[alias], 1)
                            commands[activity.command.split()[0]] += 1
                        else:
                            session.activities.remove(activity)
                    else:
                        session.activities.remove(activity)

    logging.info(
        'current load {0} with {1} unique commands from pickle'.format(sum(commands.values()), len(commands.keys())))
    known_commands = {}
    for command in set(commands.keys()) & set(manpages.keys()):
        known_commands[command] = commands[command]
    logging.info(
        'already known {0} with {1} unique commands.'.format(sum(known_commands.values()), len(known_commands.keys())))

    return sessions


def outlier_detect(sessions: [Session], lsimodel: {str: [float]}, window: int = 10, test_size: float = 0.5, folds: int = 1):
    dataset = defaultdict(list)
    for session in sessions:
        if session.userid is not None:
            for uuid, activity in sorted(session.activities.items(), key=lambda item: item[0]):
                if activity.command.split()[0] in lsimodel.keys():
                    dataset[session.userid].append(activity.command.split()[0])

    for userid, commands in dataset.items():
        dataset[userid] = list(zip(*[iter(commands)] * window))
    dataset = sorted(dataset.items(), key=lambda item: len(item[1]), reverse=True)



    for times in range(0, folds):
        for userid in dataset.keys():
            logging.debug('userid {0} has {1} commands'.format(userid, len(dataset[userid])))
            training_dataset, testing_dataset = model_selection.train_test_split(dataset[userid], test_size=test_size)





    svm.OneClassSVM()


def main():
    logging.basicConfig(level=logging.INFO)

    manpages = load_shell_commands(dumped=True)
    lsimodel = shell_commands_embedding(manpages, similar_commands=['ls', 'rm'])
    outlier_detect(load_dataset(manpages, multithread=False), lsimodel)


if __name__ == '__main__':
    main()
