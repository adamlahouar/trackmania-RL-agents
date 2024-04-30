import os
import re
import matplotlib.pyplot as plt
from numpy import mean, std

REPLAY_FOLDER_PATH = 'C:\\Users\\Adam\\Documents\\Trackmania\\Replays\\My Replays'


def get_filenames(folder_path):
    filenames: list[str] = []
    for filename in os.listdir(folder_path):
        full_path = os.path.join(folder_path, filename)
        if os.path.isfile(full_path):
            filenames.append(filename)

    return filenames


def get_time_from_filename(filename: str) -> float:
    time_string = ''.join(re.findall(r'\((.*?)\)', filename))
    parts = time_string.split('_')

    if len(parts) > 2:
        hours, minutes, seconds = parts
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)

    minutes, seconds = parts
    return int(minutes) * 60 + float(seconds)


def get_times_and_fails(filenames: list[str]) -> tuple[list[float], int]:
    times: list[float] = []
    fails = 0

    for filename in filenames:
        time = get_time_from_filename(filename)
        if time < 120:
            times.append(time)
        else:
            fails += 1

    return times, fails


def plot(times: list[float]) -> None:
    plt.hist(times, bins=15, edgecolor='black', alpha=0.8)
    plt.xlabel('Time [seconds]')
    plt.ylabel('Frequency')
    plt.title('Times')
    plt.show()


def print_statistics(times: list[float], fails: int) -> None:
    print(f'Total # of trials:  {len(times) + fails}')
    print(f'Average time:       {mean(times):.3f} seconds')
    print(f'Maximum time:       {max(times):.3f} seconds')
    print(f'Minimum time:       {min(times):.3f} seconds')
    print(f'Standard deviation: {std(times):.3f} seconds')
    print(f'Failure rate:       {100 * (fails / (len(times) + fails)):.3f}%')


if __name__ == '__main__':
    filenames = get_filenames(REPLAY_FOLDER_PATH)

    times, fails = get_times_and_fails(filenames)

    plot(times)

    print_statistics(times, fails)
