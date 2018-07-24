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
from stop_words import get_stop_words
from gensim import models, similarities, corpora

LOGPATH = ''
LOGNAME = 'log_dataset'
MANNAME = 'manual'
LSIMODEL = 'lsi.model'
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


def load_dataset(multithread: bool = False) -> [str]:
    if multithread:
        pool = []
        for period in PERIODS:
            pool.append(threading.Thread(target=read_dataset, args=(period,)))
        for thread in pool:
            thread.start()
        for thread in pool:
            thread.join()
    commands = []
    for period in PERIODS:
        logging.info('load dataset from pickle in following period: {}'.format(period))
        with open(LOGNAME + '_' + period + '.pickle', 'rb') as log_file:
            sessions = pickle.load(log_file)
        for session in sessions:
            for activity in session.activities:
                if activity.command is not None:
                    if len(activity.command.strip().split()):
                        commands.append(activity.command.strip().split()[0])
        logging.info('current load {} commands from pickle'.format(len(commands)))
    return commands


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
            man = subprocess.run(['man', command], stdout=subprocess.PIPE).stdout.decode('utf-8')
            if man:
                manpages[command] = man
        man_file = open(MANNAME + '.pickle', 'wb')
        pickle.dump(manpages, man_file)
        man_file.close()
    logging.info('read {} known shell commands'.format(len(manpages.keys())))
    return manpages


def shell_commands_embedding(manpages: {'str': 'str'}):
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

    find_similar_commands('chmod')


# # TODO: to be redo
# def parse_commands_from_local():
#     shell_commands = load_shell_commands()
#     raw_commands = []
#     for shell_command in shell_commands:
#         raw_commands.append(shell_command.raw)
#
#     with open(LOGNAME + '.txt') as f:
#         data = f.readlines()
#     dataset_commands = []
#     for line in data:
#         dataset_commands.append(line.split()[1])
#     known_commands = []
#     for dataset_command in dataset_commands:
#         if dataset_command in raw_commands:
#             known_commands.append(dataset_command)
#     logging.info('remaining {} commands known'.format(len(known_commands)))
#     return known_commands


def main():
    logging.basicConfig(level=logging.INFO)
    # load_dataset(True)
    # read_shell_commands(False)
    shell_commands_embedding(load_shell_commands(dumped=True))


if __name__ == '__main__':
    main()
