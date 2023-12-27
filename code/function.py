import numpy as np
import re
def time_difference(time1, time2):
    """
    计算两个时间的差值，时间格式为 "小时.分钟.秒"。
    返回时间差（以秒为单位）。
    """

    def time_to_seconds(time):
        h, m, s = map(int, time.split('.'))
        return h * 3600 + m * 60 + s

    if time_to_seconds(time1) > time_to_seconds(time2):
        diff = abs(time_to_seconds('24.0.0') - time_to_seconds(time1)) + time_to_seconds(time2)
    else : diff = abs(time_to_seconds(time2) - time_to_seconds(time1))
    return diff


def extract_numbers(filename):
    # 正则表达式匹配 "PN" 后面跟着任意数字，"-"，然后是一个或多个以逗号分隔的数字
    match = re.search(r'PN\d*-(\d+(?:,\d+)*)\.edf', filename)
    if match:

        return [int(num) for num in match.group(1).split(',')]
    else:

        return []


def sigma_delta_encoding(data, num_intervals, min, max):
    "计算出每个矩阵对应的阈值，比如num_intervals，就按照最大值和最小值等间隔将数值分割为num_intervals份"
    thresholds = np.linspace(min, max, num_intervals) # shape (num_intervals-1, )
    # 如果不在(min,max)做等间隔分得阈值，而是固定范围区间为(-2,6)
    # thresholds = torch.linspace(-2, 6, num_intervals+1)[1:-1]
    # print(thresholds)
    spike_list = []
    data = np.array(data)
    for line in data:
        M = np.zeros_like(data[0])
        for i in range(num_intervals-1):
            inds1 = line > thresholds[i]
            inds2 = line < thresholds[i+1]
            M[inds2*inds1] = i
        d_M = M[1:] - M[:-1]
        upper_thresh = np.where(d_M>0,d_M,0)
        upper_thresh = np.where(d_M<0,np.abs(d_M),0)
        spike_list.append(np.vstack((upper_thresh,upper_thresh)))
    return np.array(spike_list)


def print_colorful_text(text, color):
    colors = {
        'red': '\033[91m',
        'green': '\033[92m',
        'yellow': '\033[93m',
        'blue': '\033[94m',
        'magenta': '\033[95m',
        'cyan': '\033[96m',
    }
    end_color = '\033[0m'
    
    if color in colors:
        print(f"{colors[color]}{text}{end_color}")
    else:
        print(text)
        