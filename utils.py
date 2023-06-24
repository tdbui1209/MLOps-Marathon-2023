import pandas as pd
import json


def get_input(request):
    data = json.loads(request.data.decode('utf-8'))
    id = data['id']
    rows = data['rows']
    columns = data['columns']
    df = pd.DataFrame(rows, columns=columns)
    return id, df

def save_data(path, rows, columns, batch_id):
    with open(path + f"batch_{str(batch_id)}.txt", 'a') as f:
        f.write(','.join(map(str, columns)) + '\n')
        for row in rows:
            r = [str(e).replace(',', '') for e in row]
            f.write(",".join(str(e) for e in r) + '\n')