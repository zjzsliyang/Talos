import json
import talos
import logging
import collections
from flask_cors import CORS
from datetime import datetime
from talos import Session, Activity
from flask import Flask, Response, request

app = Flask(__name__)
app.debug = True
CORS(app)


# weekday_time: 0-Mon, 6-Sun
# month_time: 1-Jan, 12-Dec
basic_factors = ['all_data', 'all_commands', 'known_commands', 'periods']
summary_factors = ['weekday_time', 'month_time', 'top_commands']
other_factors = ['latest_date']
level_factors = ['dangerous', 'middle', 'slight']


def fake_data():
    all_data = 214592
    all_commands = 140472
    known_commands = 1916
    periods = 378
    sessions = None
    weekday_time = {0: 1, 1: 2, 2: 20, 3: 40, 4: 23, 5: 17, 6: 60}
    month_time = {1: 100, 2: 200, 3: 500, 4: 234, 5: 542, 6: 34, 7: 3456, 8: 236, 9: 132, 10: 1321, 11: 1345, 12: 342}
    top_commands = dict(collections.Counter({'cd': 10, 'top': 9, 'kill': 20, 'sudo': 3, 'mv': 17, 'pwd': 45, 'curl': 2, 'sh': 4, 'man': 5}).most_common(8))
    latest_date = datetime.now().strftime('%b %d, %Y')
    dangerous = [
        'kill -9 /etc/*',
        'chmod 777 -somefile',
        ' :(){ :|:& };:',
        'mv /home/user/* /dev/null']
    middle = [
        'dd if=/dev/random of=/dev/sda',
        '> file'
    ]
    slight = [
        'whoami',
        'ls',
        'ping 10.0.0.1'
    ]
    return sessions, (all_data, all_commands, known_commands, periods), (weekday_time, month_time, top_commands), (latest_date, ), (dangerous, middle, slight)

@app.before_first_request
def init():
    global info
    # manpages = talos.load_shell_commands()
    # lsimodel = talos.shell_commands_embedding(manpages)
    # sessions, basic_info, summary_info = talos.load_dataset(manpages)
    # talos.outlier_detect(sessions, lsimodel, window=1)
    sessions, basic_info, summary_info, other_info, level_info = fake_data()
    info = dict(zip(basic_factors, basic_info))
    info.update(dict(zip(summary_factors, summary_info)))
    info.update(dict(zip(other_factors, other_info)))
    info.update(dict(zip(level_factors, level_info)))
    for basic in basic_factors:
        info[basic] = format(info[basic], ',d')
    logging.info('begin service')


@app.route('/dashboard')
def dashboard():
    return Response(json.dumps(info), mimetype='application/json')


@app.route('/user', methods=['POST'])
def person():
    content = request.get_json(silent=True)
    userid = content.get('userid')

    print(userid)
    return str(userid)


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000)
