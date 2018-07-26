import os
import sys
import json
import nltk
import string
import pickle
import logging
import threading
import subprocess
from sklearn import svm
from collections import defaultdict
from stop_words import get_stop_words
from gensim import models, similarities, corpora


LOGPATH = ''
LOGNAME = 'log_dataset'
MANNAME = 'manual'
LSIMODEL = 'lsi.model'
ALIASPATH = 'alias.txt'
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
        self.activities = []
        if 'activities' in session:
            for activity in session['activities']:
                self.activities.append(Activity(activity))
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


class Shell_Command:
    def __init__(self, data: dict):
        self.raw = data['n']
        self.description = data['d']


def load_shell_commands(dumped: bool = True) -> {'str': 'str'}:
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


def shell_commands_embedding(manpages: {'str': 'str'}, similar_commands: [str] = []):
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

    def find_similar_commands(command: str):
        threshold = 0.8
        if command in commands:
            res = lsi[corpus[commands.index(command)]]
            sort_sims = sorted(enumerate(index[res]), key=lambda item: -item[1])
            print('following commands are similar to \'' + '\033[1m' + '{}'.format(command) + '\033[0m' + '\':')
            for sim in sort_sims:
                if sim[1] > threshold:
                    print('\t\033[1m' + '{}'.format(commands[sim[0]]) + '\033[0m' + ' ({})'.format(sim[1]))

    for similar_command in similar_commands:
        find_similar_commands(similar_command)


def load_dataset(manpages: {'str': 'str'}, multithread: bool = False, windows: int = 10):
    def load_alias() -> {'str': 'str'}:
        aliases = {}
        try:
            with open(ALIASPATH, 'r') as alias_file:
                for line in alias_file:
                    aliases[line.split('=', 1)[0].strip('\'')] = line.split('=', 1)[1].strip('\'').strip('\n')
        except FileNotFoundError:
            print('alias file not found, please run the command to generate: \n\talias > ' + ALIASPATH)
        return aliases

    def read_dataset(period: str):
        logging.info('reading dataset from json in following period: {}'.format(period))
        sessions = []
        for root, dirs, files in os.walk(LOGPATH + '/' + period):
            for name in files:
                file_dir = os.path.join(root, name)
                logging.debug('current read json file: {}'.format(file_dir.split('logs/')[1]))
                with open(file_dir, encoding='utf-8') as raw_log_file:
                    data = json.load(raw_log_file)
                sessions.append(Session(data))
            log_file = open(LOGNAME + '_' + period + '.pickle', 'wb')
            pickle.dump(sessions, log_file)
            log_file.close()
        logging.info('the size of log is {} MB'.format(sys.getsizeof(sessions) / 1000 / 1000))

    if multithread:
        pool = []
        for period in PERIODS:
            pool.append(threading.Thread(target=read_dataset, args=(period,)))
        for thread in pool:
            thread.start()
        for thread in pool:
            thread.join()

    aliases = load_alias()
    commands = defaultdict(int)
    for period in PERIODS:
        logging.info('load dataset from pickle in following period: {}'.format(period))
        with open(LOGNAME + '_' + period + '.pickle', 'rb') as log_file:
            sessions = pickle.load(log_file)
        for session in sessions:
            for activity in session.activities:
                if activity.command is not None:
                    activity.command = activity.command.strip()
                    if len(activity.command.split()):
                        for alias in aliases.keys():
                            if activity.command.startswith(alias + ' '):
                                activity.command = activity.command.replace(alias, aliases[alias], 1)
                        commands[activity.command.split()[0]] += 1

    logging.info(
        'current load {0} with {1} unique commands from pickle'.format(sum(commands.values()), len(commands.keys())))
    known_commands = {}
    for command in set(commands.keys()) & set(manpages.keys()):
        known_commands[command] = commands[command]
    logging.info(
        'already known {0} with {1} unique commands.'.format(sum(known_commands.values()), len(known_commands.keys())))


def main():
    logging.basicConfig(level=logging.INFO)

    manpages = load_shell_commands(dumped=True)
    shell_commands_embedding(manpages, similar_commands=['ls', 'rm'])
    load_dataset(manpages, multithread=False)


if __name__ == '__main__':
    main()
