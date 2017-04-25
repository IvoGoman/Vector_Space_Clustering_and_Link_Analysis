import datetime


def log(msg):
    print(datetime.datetime.now().strftime("%H:%M:%S.%f"), ' - ', msg)
