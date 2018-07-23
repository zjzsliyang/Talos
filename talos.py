import os
import re
import sys
import json
import nltk
import string
import pickle
import sklearn
import logging
import threading
import subprocess
from gensim import models
from stop_words import get_stop_words


LOGPATH = ''
LOGNAME = 'log_dataset'
MANNAME = 'manual'
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



def read_shell_commands(dumped: bool = True) -> {'str': 'str'}:
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


# TODO: to be redo
def parse_commands_from_local():
    shell_commands = read_shell_commands()
    raw_commands = []
    for shell_command in shell_commands:
        raw_commands.append(shell_command.raw)

    with open(LOGNAME + '.txt') as f:
        data = f.readlines()
    dataset_commands = []
    for line in data:
        dataset_commands.append(line.split()[1])
    known_commands = []
    for dataset_command in dataset_commands:
        if dataset_command in raw_commands:
            known_commands.append(dataset_command)
    logging.info('remaining {} commands known'.format(len(known_commands)))
    return known_commands


def word_embedding(known_commands: [dict]):
    with open(LOGNAME + '.txt') as man_file:
        manpages = pickle.load(man_file)
    stop_words = set(string.punctuation).union(set(get_stop_words('en')))
    for command, manual in manpages.items():
        manual.split()



    print(stop_words)
    stemmer = nltk.stem.SnowballStemmer('english')




def main():
    logging.basicConfig(level=logging.INFO)
    # parse_commands()
    # word_embedding(parse_commands_from_local())
    word_embedding()

if __name__ == '__main__':
    main()
