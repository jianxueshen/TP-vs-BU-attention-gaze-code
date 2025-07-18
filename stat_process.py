import numpy as np
import pandas as pd
from metrics import metrics

def extract_gazepoints(gp_label, gazepoints):
    if not gp_label or not gazepoints or len(gp_label) != len(gazepoints):
        return []

    current_value = gp_label[0]
    count = 0
    indices = []
    extracted_gazepoints = []
    extract_gplabel = []
    for i, value in enumerate(gp_label):
        if value == current_value:
            count += 1
            indices.append(i)
        else:
            if count > 10:
                start_index = max(0, indices[-1] - count +1 )
                sequence_gazepoints = [gazepoints[j] for j in range(start_index, indices[-1] + 1)]
                extracted_gazepoints.append(sequence_gazepoints)
                extract_gplabel.append(current_value)
            current_value = value
            count = 1
            indices = [i]

    if count > 10:
        start_index = max(0, indices[-1] - count + 1)
        sequence_gazepoints = [gazepoints[j] for j in range(start_index, indices[-1] + 1)]
        extracted_gazepoints.append(sequence_gazepoints)
        extract_gplabel.append(current_value)

    return extracted_gazepoints,extract_gplabel

def get_metrics():
    csv_file = "standfocampus/video_3.csv"
    output = "standfocampus/gt_m_video_3.csv"
    data = pd.read_csv(csv_file)
    data['isin'] = data['isin'].fillna(-1)  
    col_name = ['Gaze Point X[px]','Gaze Point Y[px]','Recording Time Stamp[ms]','Pupil Diameter Left[mm]','Pupil Diameter Right[mm]','IPD[mm]']
    gp_label = list(data['isin'])
    data = data[col_name]
    gaze =[]
    for row in data.itertuples(index=False):
        gaze.append(list(row))
    gaze_seq,seq_lab = extract_gazepoints(gp_label, gaze)
    all_metrics = []
    csv_col = []

    for i, gpinr in enumerate(gaze_seq):
        x, y, t,pl,pr,ipd = map(list, zip(*[(item[0], item[1], item[2],item[3],item[4],item[5]) for item in gpinr]))
        t0 = min(t)
        t1 = max(t)
        metrics_instance = metrics(x,y,t,pl,pr,ipd,t0,t1)
        metrics_instance_data = metrics_instance.get_metrics()

        for key, value in metrics_instance_data.items():
            if len(csv_col) < 29:
                csv_col.append(key)
            if value is None or (isinstance(value, float) and np.isnan(value)):
                metrics_instance_data[key] = 0
        feature_vector = [metrics_instance_data[key] for key in metrics_instance_data]
        feature_vector = np.array(feature_vector).reshape(1, -1)
        feature_vector = list(feature_vector[0])
        all_metrics.append(feature_vector)

    for feature,lab in zip(all_metrics,seq_lab):
        feature.append(lab)

    csv_col.append('label')
    df = pd.DataFrame(all_metrics)
    df.to_csv(output, index=False,header=csv_col)
    print(f'Saved to {output}')

if __name__ == '__main__':
    get_metrics()
    