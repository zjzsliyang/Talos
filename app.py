import collections
import json
import talos
from flask_cors import CORS
from datetime import datetime
from flask import Flask, Response

app = Flask(__name__)
app.debug = True
CORS(app)


# weekday_time: 0-Mon, 6-Sun
# month_time: 1-Jan, 12-Dec
basic_factors = ['all_data', 'all_commands', 'known_commands', 'periods']
summary_factors = ['weekday_time', 'month_time', 'top_commands']
other_factors = ['latest_date']


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
    return sessions, (all_data, all_commands, known_commands, periods), (weekday_time, month_time, top_commands), (latest_date, )

@app.before_first_request
def init():
    global info
    # manpages = talos.load_shell_commands()
    # lsimodel = talos.shell_commands_embedding(manpages)
    # sessions, basic_info, summary_info = talos.load_dataset(manpages)
    # talos.outlier_detect(sessions, lsimodel, window=1)
    sessions, basic_info, summary_info, other_info = fake_data()
    info = dict(zip(basic_factors, basic_info))
    info.update(dict(zip(summary_factors, summary_info)))
    info.update(dict(zip(other_factors, other_info)))
    for basic in basic_factors:
        info[basic] = format(info[basic], ',d')


@app.route('/dashboard')
def dashboard():
    return Response(json.dumps(info), mimetype='application/json')


if __name__ == '__main__':
    app.run(host='0.0.0.0', post=5000)
