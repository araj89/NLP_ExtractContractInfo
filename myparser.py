import pytz
from datetime import datetime

keys = ['WBC', 'BUN', 'Platelets']

year = 2020
day = '03/30'
hour = '12'

def to_float(tstr):

    tstr = tstr.replace(',', '').replace('>', '').replace('<', '')

    if tstr.isnumeric():
        return float(tstr)

    arr = tstr.split('.')
    if len(arr) != 2:
        return None

    if arr[0].isnumeric() and arr[1].isnumeric():
        return float(tstr)
    return None


#parser
def parser(f_text_to_parse, context):
    lines = []
    while True:
        line = f_text_to_parse.readline()
        if not line:
            break
        lines.append(line)

    vs = {}
    for line in lines:
        l = line.strip().replace('\t', ' ').replace('\r', ' ').replace('  ', ' ')
        words = l.split(' ')

        val = to_float(words[-1])
        if val == None:
            continue
        if not words[0] in keys:
            continue
        vs[words[0]] = val

    res = []

    #from datetime import timezone

    ts = datetime.strptime(f'{year}/{day}/{hour}', '%Y/%m/%d/%H')
    ts = ts.replace(tzinfo=pytz.timezone(context['config']['hospital timezone']))
    utc_dt = ts.astimezone(pytz.utc)
    #ts = ts.replace(tzinfo=timezone.utc).timestamp()


    for k, v in vs.items():
        res.append({'data': {
            'mmt' : k,
            'rt' : datetime.now(),
            'ts' : utc_dt,
            'val' : v,
        }, 'type' : 'measurement'})

    return res

if __name__ == "__main__":
    context = {'config': {'hospital timezone': 'America/New_York'}}
    path = 'C:\\Users\\ARAJ\\Downloads\\Dascena Programming Test\\Dascena Programming Test\\labs5-d.txt'
    f = open(path, encoding='utf-8')
    res = parser(f, context)
    print (res)
    f.close()





