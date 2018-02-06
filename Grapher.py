import matplotlib.pyplot as plt
import numpy, traceback
import csv, os, sys, math
from configparser import ConfigParser

class Subject_Data:
  DATE_FORMAT = '%d/%m/%y'
  SUBJECT_DATA_PREFIX = 'Sensor Data '

  def __init__(self, folder_name):
    self.folder_name = folder_name
    self.log_file = folder_name + '/log.txt'
    self.data = { 'calibration1' : None, 'calibration2' : None, 'trial' : None}
    self.calibration_problem = None
    self.process_log_file()
    self.calibration_data = {}

  def add_sample_data(self, data_name, sample_data):
    self.data[data_name] = sample_data

  def get_sample_data(self, data_name):
    return self.data[data_name]

  def process_log_file(self):
    self.log_data = {}
    with open(self.log_file, 'r') as f:
      for line_data in f:
        if line_data.strip() != '':
          data = [x.strip() for x in line_data.split(':')]
          self.log_data.update({data[0]: data[1]})

  def generate_calibration_data(self):
    for period in ['calibration1', 'calibration2']:
      if self.log_data[period] != 'NA':
        self.calibration_data.update({ period : {}})
        for digit in ['index', 'thumb']:
          if digit == 'index':
            self.calibration_data[period].update({ digit : CalibrationData(self.data[period], 1, 0) })
            self.calibration_data[period][digit].configure(6, 2, 10, 30, 5)
          else:
            self.calibration_data[period].update({ digit : CalibrationData(self.data[period], 2, 0) })
            self.calibration_data[period][digit].configure(26, 2, 10, 30, 5)
          self.calibration_data[period][digit].generate_sample_set()

  def print_calibration_data(self):
    for period in ['calibration1', 'calibration2']:
      for digit in ['index', 'thumb']:
        print('%s, %s: %s' % (digit, period, self.calibration_data[period][digit]))

  def __str__(self):
    # output_str = ''
    # for k, v in self.log_data.items():
    #   output_str += '%s: %s\n' % (k, v)
    # return output_str
    return '%s - %s' % (self.folder_name, self.calibration_problem)

class SampleData:
  timestamp_value = 0
  def __init__(self, file_name, index_col=1, thumb_col=2, freq=10):
    SampleData.timestamp_value = 0
    self.file_name = file_name
    self.data = []
    with open(file_name, 'r') as f:
      csv_reader = csv.reader(f, delimiter=',', quotechar='"')
      for line in csv_reader:
        try:
          self.data.append(SampleDataEntry(line, index_col, thumb_col, SampleData.timestamp_value))
          SampleData.timestamp_value += 1/freq
        except IndexError:
          print('Error Creating Sample Data Entry')
          print('file_name = %s' % file_name)
          print('index_col = %d' % index_col)
          print('thumb_col = %d' % thumb_col)
          print('frequency = %d' % freq)
          print('line = %s' % line)
          sys.exit(1)

  def get_timedata(self):
    timedata = []
    for d in self.data:
      timedata.append(d.time)
    return timedata

  def get_pos_data(self, digit):
    digit_data = []
    for d in self.data:
      digit_data.append(d.index_pos if digit == 'index' else d.thumb_pos)
    return digit_data

  def get_average(self, digit, start_point=None, end_point=None):
    if start_point is None:
      start_point = 0
    if end_point is None:
      end_point = len(self.data)
    total = 0
    for d in self.data[start_point:end_point]:
      total += d.get_value(digit)
    return total/(end_point - start_point)

  def get_max(self, digit):
    values = []
    for d in self.data:
      values.append(d.get_value(1 if digit == 'index' else 2))
    return numpy.amax(values)

  def get_min(self, digit):
    values = []
    for d in self.data:
      values.append(d.get_value(1 if digit == 'index' else 2))
    return numpy.amin(values)

class SampleDataEntry:
  def __init__(self, file_line, index_col, thumb_col, time_val):
    self.index_pos = float(file_line[index_col])
    self.thumb_pos = float(file_line[thumb_col])
    self.time = time_val

  def get_value(self, col):
    if col == 0:
      return self.time
    elif col == 1:
      return self.index_pos
    elif col == 2:
      return self.thumb_pos
    else:
      return None

class CalibrationData:
  def __init__(self, sample_data, column, time_col):
    self.sample_data = sample_data
    self.column = column
    self.time_col = time_col
    self.sample_points = []
    self.sample_set = []

  def configure(self, offset, sample_size, frequency, step_size, sample_count):
    for i in range(sample_count):
      self.sample_points.append(
        [frequency * (offset + step_size * i), frequency * (offset + step_size * i + sample_size)])

  def generate_sample_set(self):
    for sp in self.sample_points:
      # print(self.file_data.data[sp[0]].get_value('timestamp'))
      self.sample_set.append(self.sample_data.get_average(self.column, sp[0], sp[1]))

  def get_average(self):
    return numpy.mean(self.sample_set)

  def get_min(self):
    return numpy.amin(self.sample_set)

  def get_max(self):
    return numpy.amax(self.sample_set)

  def __str__(self):
    return 'max = %0.2f, min = %0.2f, average = %0.2f' % (self.get_max(), self.get_min(), self.get_average())

def main():
  config_file = 'stroke_config.cfg'
  config_data = ConfigParser()
  config_data.read(config_file)
  sample_data = {}

  for s in config_data.sections():
    sample_data.update({ s : Subject_Data(config_data.get(s, 'directory')) })
    calibration_file1 = '%s/%s%s.csv' % (config_data.get(s, 'directory'), Subject_Data.SUBJECT_DATA_PREFIX, sample_data[s].log_data['calibration1'])
    calibration_file2 = '%s/%s%s.csv' % (config_data.get(s, 'directory'), Subject_Data.SUBJECT_DATA_PREFIX, sample_data[s].log_data['calibration2'])
    trial_file = '%s/%s%s.csv' % (config_data.get(s, 'directory'), Subject_Data.SUBJECT_DATA_PREFIX, sample_data[s].log_data['trial'])
    sample_data[s].add_sample_data('calibration1', SampleData(calibration_file1, config_data.getint(s, 'index'), config_data.getint(s, 'thumb'), config_data.getint(s, 'frequency')))
    sample_data[s].add_sample_data('trial', SampleData(trial_file, config_data.getint(s, 'index'), config_data.getint(s, 'thumb'), config_data.getint(s, 'frequency')))
    if  sample_data[s].log_data['calibration2'] != 'NA':
      sample_data[s].add_sample_data('calibration2', SampleData(calibration_file2, config_data.getint(s, 'index'), config_data.getint(s, 'thumb'), config_data.getint(s, 'frequency')))
    sample_data[s].generate_calibration_data()

  bad_config_data = []

  for digit in ['index', 'thumb']:
    max_value = None
    min_value = None

    for k,v in sample_data.items():
      sample_min = []
      sample_max = []

      sample_min.append(v.calibration_data['calibration1'][digit].get_min())
      sample_max.append(v.calibration_data['calibration1'][digit].get_max())

      if v.log_data['calibration2'] != 'NA':
        sample_min.append(v.calibration_data['calibration2'][digit].get_min())
        sample_max.append(v.calibration_data['calibration2'][digit].get_max())

      # if len(sample_min) == 2 and len(sample_max) == 2:
      #   if abs(sample_min[0] - sample_min[1])/sample_min[0] > 0.2:
      #     bad_config_data.append(v)
      #     v.calibration_problem = 'Min %s Varyance too great' % (digit.capitalize())
      #     continue
      #   if abs(sample_max[0] - sample_max[1])/sample_max[0] > 0.2:
      #     bad_config_data.append(v)
      #     v.calibration_problem = 'Max %s Varyance too great' % (digit.capitalize())
      #     continue
      #   if abs(sample_min[0] - sample_max[0]) > 100:
      #     bad_config_data.append(v)
      #     v.calibration_problem = 'Min %s Difference too great' % (digit.capitalize())
      #     continue
      #   if abs(sample_min[1] - sample_max[1]) > 100:
      #     bad_config_data.append(v)
      #     v.calibration_problem = 'Max %s Difference too great' % (digit.capitalize())
      #     continue
      #   if sample_min[0] < 200 or sample_min[1] < 200:
      #     bad_config_data.append(v)
      #     v.calibration_problem = 'Min %s Value too low' % (digit.capitalize())
      #     continue

      plt.legend()
      plt.plot(10, sample_min[0], 'bo')
      plt.plot(10, sample_max[0], 'ro')
      if len(sample_min) == 2:
        plt.plot(20, sample_min[1], 'bo')
        plt.plot(20, sample_max[1], 'ro')
        plt.plot([10, 20], [sample_min[0], sample_min[1]], 'k-')
        plt.plot([10, 20], [sample_max[0], sample_max[1]], 'k-')

      for i in range(len(sample_min)):
        if min_value is None or min_value > sample_min[i]:
          min_value = sample_min[i]
      for i in range(len(sample_max)):
        if max_value is None or max_value < sample_max[i]:
          max_value = sample_max[i]


      # plt.plot([10, v.calibration_data['calibration1']['index'].get_min()], [20, v.calibration_data['calibration2']['index'].get_min()], 'k-')
      # plt.plot([10, v.calibration_data['calibration1']['index'].get_max()], [20, v.calibration_data['calibration2']['index'].get_max()], 'k-')
    # plt.plot([10, 20], [200, 250], 'k-')
    plt.ylim(math.floor(min_value/10)*10, math.ceil(max_value/10)*10)
    plt.yticks(numpy.arange(math.floor(min_value/10)*10, math.ceil(max_value/10)*10+1, 50))
    plt.xlim(5, 25)
    plt.xticks(numpy.arange(10, 21, 10), ('Before', 'After'))
    plt.savefig('graphs/%sCalibration.png' % (digit.capitalize()))
    plt.close()

  for bcd in bad_config_data:
    print(bcd)
  # print(bad_config_data)

  for test in ['CON1', 'CON13', 'STR4', 'STR5', 'STR11']:
    for cali in ['calibration1', 'calibration2']:
      for digit in ['thumb', 'index']:
        subject = sample_data[test]
        sd = subject.get_sample_data(cali)
        plt.plot(sd.get_timedata(), sd.get_pos_data(digit))
        min_value = sd.get_min(digit)
        max_value = sd.get_max(digit)
        plt.yticks(numpy.arange(min_value, max_value + 1, 10))
        plt.savefig('%s/%s - %s %s.png' % ('graphs', test, cali.capitalize(), digit.capitalize()))
        plt.close()


  #
  # fig, ax = plt.subplots()
  # ax.plot(sample_data['CON1'].get_sample_data('calibration1').get_timedata(), sample_data['CON1'].get_sample_data('calibration1').get_pos_data('index'))
  # ax.set(xlabel='Time', ylabel='Position')
  # # ax.grid()
  # plt.show()
  # fig, ax = plt.subplots()
  # ax.plot(sample_data['CON2'].get_sample_data('calibration1').get_timedata(), sample_data['CON2'].get_sample_data('calibration1').get_pos_data('index'))
  # ax.set(xlabel='Time', ylabel='Position')
  # plt.show()
  #

if __name__ == '__main__':
  main()