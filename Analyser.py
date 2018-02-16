import csv, os, numpy, sys, math
from configparser import ConfigParser
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np

class Subject:
  """
  The Subject class is created for each subject/patient in the study.
  """
  DATA_FILE_PREFIX = 'Sensor Data '
  def __init__(self, folder_name, settings):
    self.folder_name = folder_name
    self.log_file = '%s/log.txt' % self.folder_name
    self.data_set = DataSet()
    self.log_data = {}
    self.process_logfile()
    self.calibration_data = {}
    self.digit_list = settings['digits'].copy()
    self.max_flexion = {}
    self.max_extension = {}
    for d in self.digit_list:
      self.max_flexion.update( { d : 0 })
      self.max_extension.update( { d : 0 })
    self.position_histogram = None
    for i in ['calibration1', 'calibration2']:
      self.calibration_data.update({ i : {} })
      for j in self.digit_list:
        self.calibration_data[i].update({ j : {} })
        for k in ['min', 'max']:
          self.calibration_data[i][j].update({ k : None })
    self.movement_groupings = {}
    self.velocity_groupings = {}
    self.average_amplitude = { x : 0 for x in self.digit_list }
    self.total_amplitude = { x : 0 for x in self.digit_list }
    self.movement_count = { x : 0 for x in self.digit_list }
    self.longest_idle_time = { x : 0 for x in self.digit_list }
    self.total_idle_time = { x : 0 for x in self.digit_list }
    self.average_movement = { x : 0 for x in self.digit_list }
    self.average_velocity = { x : 0 for x in self.digit_list }
    self.ROM = { x : 0 for x in self.digit_list }

  def process_logfile(self):
    """
    Reads the log file (log.txt) in the directory and stores all of the information in an dict
    :return:
    """
    with open(self.log_file, 'r') as f:
      for line in f:
        if line.strip() != '':
          data = [x.strip() for x in line.split(':')]
          self.log_data.update({ data[0] : data[1] })

  def write_log_to_csv(self, csv_file):
    """
    Writes the data from the log file (log.txt) to a csv file
    :param csv_file: The csv file to populate
    :return:
    """
    output_array = []
    output_array.append(self.log_data['sub'])
    output_array.append(0 if self.log_data['sub'].startswith('CON') else 1)
    output_order = ['date', 'age', 'sex', 'stroke_date', 'stroke_type', 'side_tested', 'mas_score7', 'mas_score8', 'hand_dom']
    for f in output_order:
      if f in self.log_data:
        output_array.append(self.log_data[f])
      else:
        output_array.append('NA')
    csv_file.writerow(output_array)

  def generate_output_file_prefix(self):
    """
    Generates the prefix of a csv file with common data
    :return:
    """
    output_array = [self.log_data['sub'], 0 if self.log_data['sub'].startswith('CON') else 1, self.log_data['age'], self.log_data['sex']]
    return output_array

  def read_sample_file(self, data_type, index_col, thumb_col, freq):
    """
    Reads the contents of a file into the data_set object
    :param data_type: The type of file being read, calibration1, calibration2 or trial
    :param index_col: The column that contains the index finger position data
    :param thumb_col: The column that contains the thumb position data
    :param freq: The sample frequency
    :return:
    """
    if data_type in self.log_data:
      file_name = '%s/%s%s.csv' % (self.folder_name, Subject.DATA_FILE_PREFIX, self.log_data[data_type])
      if os.path.isfile(file_name):
        print('%s - Starting Processing %s' % (datetime.now().strftime('%H:%M:%S'), file_name))
        self.data_set.add_data(data_type, file_name, index_col, thumb_col, freq)
        print('%s - Finished Processing %s' % (datetime.now().strftime('%H:%M:%S'), file_name))
      else:
        print('File %s does not exist' % file_name)

  def generate_calibration_data(self, calibration_type, digit, min_max, frequency, offset, sample_size, step_size, sample_count):
    """
    Processes the Calibration data and populates the maximum and minumum flexion/extension
    :param calibration_type: calibration1 or calibration2
    :param digit: finger or thumb
    :param min_max: whether this is to determine a minimum value or max value
    :param frequency: The sample rate
    :param offset: Time to ignore prior to taking the calibration sample
    :param sample_size: Time length of the sample
    :param step_size: Time interval between samples
    :param sample_count: the number of samples to take
    :return:
    """
    if self.data_set.data[calibration_type] is None:
      return
    calibration = CalibrationData(self.data_set.data[calibration_type], digit)
    calibration.configure(frequency, offset, sample_size, step_size, sample_count)
    try:
      calibration.generate_sample_set()
    except ZeroDivisionError:
      print('Zero Division Error')
      print('calibration_type = %s' % calibration_type)
      print('digit = %s' % digit)
      print('min_max = %s' % min_max)
      sys.exit(1)
    if min_max == 'min':
      self.calibration_data[calibration_type][digit][min_max] = calibration.get_min()
    else:
      self.calibration_data[calibration_type][digit][min_max] = calibration.get_max()

    fig = plt.figure(figsize=(15, 7.5))
    sp = fig.add_subplot(1, 1, 1)
    sp.set_xlim(0, 6)
    sp.set_xlabel('Calibration')
    sp.set_xticks([1, 2, 3, 4, 5])
    sp.set_ylabel('Arbitary Units')
    calibration.add_to_plot(sp)
    fig.savefig('graphs/%s_%s_%s_%s_calibration_points.png' % (self.folder_name, calibration_type, digit, min_max))
    plt.close()

    fig = plt.figure(figsize=(20, 10))
    sp = fig.add_subplot(1, 1, 1)
    sp.set_xlim(0, self.data_set.data[calibration_type].data[-1].timestamp)
    sp.set_xlabel('Time (s)')
    sp.set_ylabel('Arbitary Units')
    plot_time_data, plot_position_data = self.data_set.data[calibration_type].get_plot_data(digit)
    sp.plot(plot_time_data, plot_position_data, 'k')
    for sample_point in calibration.sample_points:
      plot_time_data, plot_position_data = self.data_set.data[calibration_type].get_plot_data(digit, sample_point[0]/frequency, sample_point[1]/frequency)
      sp.plot(plot_time_data, plot_position_data, 'r', linewidth=2)
    fig.savefig('graphs/%s_%s_%s_%s_position_data_with_calibration.png' % (self.folder_name, calibration_type, digit, min_max))
    plt.close()




  def generate_calibration_plot(self):
    """
    Generates a matplotlib of the min and max values of each digit showing their calibration values from the two calibration file
    :return:
    """
    for d in self.digit_list:
      sp = plt.subplot(1, 1, 1)
      sp.set_xlim(5, 25)
      sp.set_xlabel('Calibration')
      sp.set_xticks([10, 20])
      sp.set_xticklabels(['calibration1', 'calibration2'])
      sp.set_ylabel('Arbitary Units')
      self.add_calibration_data_to_plot(sp, d)
      plt.savefig('graphs/%s_%s.png' % (self.folder_name, d))
      plt.close()

  def add_calibration_data_to_plot(self, sp, d):
    """
    Adds the calibration data to an existing plot
    :param sp: The plot
    :param d: the digit to plot
    :return:
    """
    sp.plot(10, self.calibration_data['calibration1'][d]['min'], 'bo', markersize=4)
    sp.plot(10, self.calibration_data['calibration1'][d]['max'], 'ro', markersize=4)
    if self.data_set.data['calibration2'] is not None:
      sp.plot(20, self.calibration_data['calibration2'][d]['min'], 'bo')
      sp.plot(20, self.calibration_data['calibration2'][d]['max'], 'ro')
      sp.plot([10, 20], [self.calibration_data['calibration1'][d]['min'], self.calibration_data['calibration2'][d]['min']], 'k-', linewidth=0.5)
      sp.plot([10, 20], [self.calibration_data['calibration1'][d]['max'], self.calibration_data['calibration2'][d]['max']], 'k-', linewidth=0.5)

  def produce_position_plot(self, data_source, digit):
    """
    Produces a position plot of the data
    :param data_source: calibration1, calibration2 or trial
    :param digit: thumb or index
    :return:
    """
    if self.data_set.data[data_source] is None or digit not in self.digit_list:
      return
    fig = plt.figure(figsize=(20, 10))
    sp = fig.add_subplot(1, 1, 1)
    sp.set_xlim(0, self.data_set.data[data_source].data[-1].timestamp)
    sp.set_xlabel('Time (s)')
    sp.set_ylabel('Arbitary Units')
    plot_time_data, plot_position_data = self.data_set.data[data_source].get_plot_data(digit)
    sp.plot(plot_time_data, plot_position_data)
    fig.savefig('graphs/%s_%s_%s_position_data.png' % (self.folder_name, data_source, digit))
    plt.close()

  def calc_calibration_values(self):
    """
    Calculates the various calibration values necessary for data analysis.
    The key values calculated here are for the ROM, the max flexion and extension for each digit.
    :return:
    """
    cali_data = {}
    for c in ['calibration1', 'calibration2']:
      for d in self.digit_list:
        if not d in cali_data:
          cali_data.update( { d : {} })
        for m in ['min', 'max']:
          if not m in cali_data[d]:
            cali_data[d].update( { m : []})
          cali_data[d][m].append(self.calibration_data[c][d][m])
    for d in self.digit_list:
      # There may be only one calibration file, if there is, get the max/min from that, not an average of both
      if None in cali_data[d]['max']:
        self.max_flexion[d] = cali_data[d]['max'][0] if cali_data[d]['max'][1] is None else cali_data[d]['max'][1]
      else:
        self.max_flexion[d] = numpy.mean(cali_data[d]['max'])
      if None in cali_data[d]['min']:
        self.max_extension[d] = cali_data[d]['min'][0] if cali_data[d]['min'][1] is None else cali_data[d]['min'][1]
      else:
        self.max_extension[d] = numpy.mean(cali_data[d]['min'])
      self.ROM[d] = self.max_flexion[d] - self.max_extension[d]

  def calc_histogram_plot(self, bucket_count):
    """
    Produce histogram data of the times spent in various positions
    :param bucket_count: The number of groupings
    :return:
    """
    self.position_histogram = {}
    for d in self.digit_list:
      self.position_histogram.update( { d : None })
    for d in self.digit_list:
      self.position_histogram[d] = Histogram(self.max_extension[d], self.max_flexion[d], bucket_count)
      self.position_histogram[d].process_data(self.data_set, d)
      self.position_histogram[d].calc_percentages()
      # print(d)
      # print(self.position_histogram[d])

  def write_histogram_to_file(self, subject_name, csv_file_raw, csv_file_percentage):
    """
    Writes the Histogram Data to the specified file
    :param subject_name: The name of the subject
    :param csv_file_raw: the file object that will store the raw numbers
    :param csv_file_percentage: the file object that will store the percentages of all time
    :return:
    """
    for d in self.digit_list:
      output_array = [subject_name, d]
      self.position_histogram[d].write_to_file(output_array, csv_file_raw)
      output_array = [subject_name, d]
      self.position_histogram[d].write_to_file(output_array, csv_file_percentage, False)

  def generate_velocity_change_plots(self):
    """
    Generates the Velocity Change data for each data file
    :return:
    """
    for i in ['calibration1', 'calibration2', 'trial']:
      if self.data_set.data[i] is not None:
        self.data_set.data[i].generate_velocity_change_plot('graphs/' + self.folder_name + '_%s_' + i + '.png')

  def produce_histograms(self, movement_buckets, velocity_buckets, time_block, min_threshold=0):
    """
    Populates Histogram groupings based on movement and velocity groupings as well as breaking the trial period into chunks
    :param movement_buckets: Array (sorted) containing the movement data to determine the cutoffs for each grouping
    :param velocity_buckets: Array (sorted) containing the velocity data to determine the cutoffs for each grouping
    :param time_block: Time (in seconds) to break the trial data into
    :param min_threshold: The minimum threshold of movement to determine if it is a valid movement
    :return:
    """
    end_time = self.data_set.data['trial'].data[-1].timestamp
    for i in self.digit_list:
      self.movement_groupings.update( { i : [0] * (len(movement_buckets[i])+1) } )
      self.velocity_groupings.update( { i : [0] * (len(velocity_buckets[i])+1) } )
      for t in range(time_block, math.ceil(end_time), time_block):
        sample_start_time = (t - time_block)
        sample_end_time = t
        self.populate_movement_groupings(i, movement_buckets, sample_start_time, sample_end_time, min_threshold)
        self.populate_velocity_groupings(i, velocity_buckets, sample_start_time, sample_end_time, min_threshold)
      self.average_movement[i] = self.data_set.data['trial'].get_movement_count(i, 0, end_time, min_threshold)/end_time
      self.average_velocity[i] = self.data_set.data['trial'].get_movement_average_velocity(i, 0, end_time, min_threshold)


  def populate_movement_groupings(self, digit, buckets, start_time, end_time, min_threshold):
    """
    Populates the movement groupings data with the movement rate over the given time period (between start_time and end_time)
    :param digit: The digit (finger or thumb) to collect data for
    :param buckets: The thresold limits
    :param start_time: The start time (seconds) of the block to analyse
    :param end_time:  The end time (seconds) of the block to analyse
    :param min_threshold: The minimum threshold of any movement
    :return:
    """
    movement_rate = self.data_set.data['trial'].get_movement_count(digit, start_time, end_time, min_threshold)/(end_time - start_time)
    added_to_array = False
    for i in range(len(buckets[digit])):
      if movement_rate < buckets[digit][i]:
        self.movement_groupings[digit][i] += 1
        added_to_array = True
        break
    if not added_to_array:
      self.movement_groupings[digit][-1] += 1

  def populate_velocity_groupings(self, digit, buckets, start_time, end_time, min_threshold=0):
    """
    Populates the velocity groupings data with the movement rate over the given time period (between start_time and end_time)
    :param digit: The digit (finger or thumb) to collect data for
    :param buckets: The thresold limits
    :param start_time: The start time (seconds) of the block to analyse
    :param end_time:  The end time (seconds) of the block to analyse
    :param min_threshold: The minimum threshold of any movement
    :return:
    """
    average_velocity = self.data_set.data['trial'].get_movement_average_velocity(digit, start_time, end_time, min_threshold)/self.ROM[digit]
    added_to_array = False
    for i in range(len(buckets[digit])):
      if average_velocity < buckets[digit][i]:
        self.velocity_groupings[digit][i] += 1
        added_to_array = True
        break
    if not added_to_array:
      self.velocity_groupings[digit][-1] += 1

  def get_average_movement_amplitude(self, min_threshold):
    """
    Calculates the average amplitude of movement over the entire trial period
    :param min_threshold: The minimum movement to consider as valid
    :return:
    """
    for d in self.digit_list:
      self.total_amplitude[d], self.movement_count[d], self.average_amplitude[d] = self.data_set.data['trial'].get_movement_average_magnitude(d, 0, self.data_set.data['trial'].data[-1].timestamp, min_threshold)

  def get_idle_times(self, min_threshold):
    """
    Calculate the longest and total idle time of all digits for the trial
    :param min_threshold: The minimum movement to consider as valid
    :return:
    """
    for d in self.digit_list:
      self.longest_idle_time[d], self.total_idle_time[d] = self.data_set.data['trial'].calc_idle_time(d, 0, self.data_set.data['trial'].data[-1].timestamp, min_threshold)

  def write_summary(self, settings):
    """
    Produces a summary file with all of the calculated data for the subject/patient
    :param settings: The settings data from config.ini
    :return:
    """
    with open('%s/%s_statistical_analysis_summary.csv' % (self.folder_name, self.folder_name), 'w') as f:
      csv_writer = csv.writer(f, delimiter=',', quotechar='"')
      csv_writer.writerow(['Patient', self.folder_name])
      csv_writer.writerow(['Analysed Digits', ', '.join(self.digit_list)])
      csv_writer.writerow(['Movement Threshold', settings['movement_threshold']])
      csv_writer.writerow(['Segment Time (s)', settings['segment_time']])
      csv_writer.writerow('')

      for d in self.digit_list:
        csv_writer.writerow(['%s Average Amplitude' % d.capitalize(), '%0.2f' % self.average_amplitude[d]])

      csv_writer.writerow('')

      for d in self.digit_list:
        csv_writer.writerow(['%s Calibration Max Flexion' % d.capitalize(), '%0.2f' % self.max_flexion[d] ])
        csv_writer.writerow(['%s Calibration Max Extension' % d.capitalize(), '%0.2f' % self.max_extension[d] ])
        csv_writer.writerow(['%s ROM' % d.capitalize(), '%0.2f' % self.ROM[d]])

      csv_writer.writerow('')
      for d in self.digit_list:
        csv_writer.writerow(['%s Average Movements/Sec' % d.capitalize(), '%0.2f' % self.average_movement[d]])
        csv_writer.writerow(['%s Movement Histogram Data' % d.capitalize()])
        output_array = []
        thresholds = settings['movement_count_thresholds_%s' % d]
        output_array.append('< %0.2f' % thresholds[0])
        for i in range(len(thresholds) - 1):
          output_array.append('%0.2f - %0.2f' % (thresholds[i], thresholds[i+1]))
        output_array.append('> %0.2f' % thresholds[-1])
        csv_writer.writerow(output_array)
        csv_writer.writerow(self.movement_groupings[d])
        csv_writer.writerow('')

      csv_writer.writerow('')
      for d in self.digit_list:
        csv_writer.writerow(['%s Average Velocity' % d.capitalize(), '%0.2f' % (self.average_velocity[d]/self.ROM[d])])
        csv_writer.writerow(['%s Average Velocity Histogram Data' % d.capitalize()])
        output_array = []
        thresholds = settings['velocity_thresholds_%s' % d]
        output_array.append('< %0.2f' % thresholds[0])
        for i in range(len(thresholds) - 1):
          output_array.append('%0.2f - %0.2f' % (thresholds[i], thresholds[i+1]))
        output_array.append('> %0.2f' % thresholds[-1])
        csv_writer.writerow(output_array)
        csv_writer.writerow(self.velocity_groupings[d])
        csv_writer.writerow('')

      csv_writer.writerow('')
      for d in self.digit_list:
        csv_writer.writerow(['%s Idle Time' % d.capitalize(), '%0.2f%%' % ((self.total_idle_time[d]/self.data_set.data['trial'].data[-1].timestamp)*100)])
        csv_writer.writerow(['%s Longest Idle Time (s)' % d.capitalize(), '%0.2f' % self.longest_idle_time[d]])

      csv_writer.writerow('')
      output_array = ['Digit', '< 0']
      for i in range(settings['percentile_count']):
        output_array.append('%d - %d' % (i * settings['percentile_count'], (i + 1) * settings['percentile_count']))
      output_array.append('> 100')
      csv_writer.writerow(output_array)
      for d in self.digit_list:
        output_array = [d.capitalize()]
        self.position_histogram[d].write_to_file(output_array, csv_writer)
        output_array = [d.capitalize()]
        self.position_histogram[d].write_to_file(output_array, csv_writer, False)
    self.data_set.data['trial'].print_velocity_data('%s/%s_Velocity_Data.csv' % (self.folder_name, self.folder_name))

class Histogram:
  """
  Class to hold histogram data
  """
  def __init__(self, min, max, bucket_count):
    self.min = min
    self.max = max
    self.bucket_count = bucket_count
    self.buckets = []
    self.total_entries = 0
    step_size = (max - min)/bucket_count
    for i in range(bucket_count):
      self.buckets.append(HistogramBucket(min + i*step_size, min + (i+1)*step_size))
    # Contains buckets for values that are beyond the upper and lower bounds
    self.beyond_upper = HistogramBucket(max, 0)
    self.beyond_lower = HistogramBucket(0, min)

  def process_data(self, dataset, digit):
    """
    Adds the trial data into the histogram buckets
    :param dataset: The dataset to analyse
    :param digit: the digit to process
    :return:
    """
    for p in dataset.data['trial'].data:
      if digit == 'thumb':
        position = p.thumb_pos
      else:
        position = p.index_pos
      if self.beyond_lower.add_to_bucket(position):
        self.total_entries += 1
        continue
      if self.beyond_upper.add_to_bucket(position):
        self.total_entries += 1
        continue
      for b in self.buckets:
        if b.add_to_bucket(position):
          self.total_entries += 1
          break

  def calc_percentages(self):
    """
    Calculates the percentages of total values for each bucket
    :return:
    """
    self.beyond_lower.calc_percentage(self.total_entries)
    for b in self.buckets:
      b.calc_percentage(self.total_entries)
    self.beyond_upper.calc_percentage(self.total_entries)

  def get_percentages_array(self):
    """
    Produces an array of the bucket percentages
    :return: percentages array
    """
    tmp_array = [self.beyond_lower.percentage]
    for b in self.buckets:
      tmp_array.append(b.percentage)
    tmp_array.append(self.beyond_upper.percentage)
    return tmp_array

  def write_to_file(self, output_array, csv_file, as_raw=True):
    """
    Writes the bucket data to file
    :param output_array: The existing array of data
    :param csv_file: The csv file object to write to
    :param as_raw: Boolean: Whether the data is exported as raw number or as percentages
    :return:
    """
    if as_raw:
      output_array.append(self.beyond_lower.count)
    else:
      output_array.append('%0.2f%%' % self.beyond_lower.percentage)
    for b in self.buckets:
      if as_raw:
        output_array.append(b.count)
      else:
        output_array.append('%0.2f%%' % b.percentage)
    if as_raw:
      output_array.append(self.beyond_upper.count)
    else:
      output_array.append('%0.2f%%' % self.beyond_upper.percentage)
    csv_file.writerow(output_array)

  def __str__(self):
    """
    Writes the data to a string
    :return:
    """
    self.beyond_lower.calc_percentage(self.total_entries)
    outputstr = '%s\n' % self.beyond_lower
    for b in self.buckets:
      b.calc_percentage(self.total_entries)
      outputstr += '%s\n' % b
    self.beyond_upper.calc_percentage(self.total_entries)
    outputstr += '%s\n' % self.beyond_upper
    return outputstr[:-1]

class HistogramBucket:
  """
  The class to hold bucket data for the histogram class
  """
  def __init__(self, lower_bound, upper_bound):
    """
    :param lower_bound: The lower limit
    :param upper_bound: the upper limit
    """
    self.lower_bound = lower_bound
    self.upper_bound = upper_bound
    self.count = 0
    self.percentage = 0

  def add_to_bucket(self, value):
    """
    If the given value is within the range and if so adds it to the bucket
    :param value: The value to compare and add
    :return: Boolean: True the value was added, False, it was not added
    """
    if value < self.upper_bound and value >= self.lower_bound:
      self.count += 1
      return True
    if self.lower_bound == 0 and value <= self.upper_bound:
      self.count += 1
      return True
    if self.upper_bound == 0 and value > self.lower_bound:
      self.count += 1
      return True
    return False

  def calc_percentage(self, total_entries):
    """
    Calculate the percentage of the values in the bucket compared to the total number of entries
    :param total_entries:
    :return:
    """
    self.percentage = self.count/total_entries * 100

  def __str__(self):
    if self.upper_bound == 0:
      return '> %0.2f: %d - %0.2f%%' % (self.lower_bound, self.count, self.percentage)
    if self.lower_bound == 0:
      return '< %0.2f: %d - %0.2f%%' % (self.upper_bound, self.count, self.percentage)
    return '%0.2f - %0.2f: %d - %0.2f%%' % (self.lower_bound, self.upper_bound, self.count, self.percentage)

class DataSet:
  """
  Holds the dataset for the subject (probably a useless class as it's really only a dict)
  """
  def __init__(self):
    self.data = { 'trial' : None, 'calibration1' : None, 'calibration2' : None }

  def add_data(self, data_type, file_name, index_col=1, thumb_col=2, freq=10):
    """
    Adds the data to the object
    :param data_type: string: calibration1, calibration2, trial
    :param file_name: The file to read in
    :param index_col: The column containing the index position data
    :param thumb_col: The column containing the thumb position data
    :param freq:
    :return:
    """
    if data_type in self.data:
      self.data[data_type] = SampleData(file_name, index_col, thumb_col, freq)

class SampleData:
  """
  Holds the data for the relevant file
  """
  def __init__(self, file_name, index_col=1, thumb_col=2, freq=10):
    self.file_name = file_name
    self.index_col = index_col
    self.thumb_col = thumb_col
    self.freq = freq
    self.data = []
    self.velocity_change_data = {'index' : [], 'thumb' : []}
    self.process_file()

  def get_movement_count(self, digit, start_time=None, end_time=None, threshold=0):
    """
    Calculates the number of movements that have occured within the specified time period
    :param digit: index or thumb, the digit to calculate
    :param start_time: time in seconds to begin the analysis for
    :param end_time: time in seconds to end the analysis for
    :param threshold: the smallest movement which will be considered as valid
    :return: The number of movements
    """
    if digit not in self.velocity_change_data:
      return
    if start_time is None:
      start_time = 0
    if end_time is None:
      end_time = self.velocity_change_data[digit][-1].end_time
    movement_count = 0
    for v in self.velocity_change_data[digit]:
      if v.start_time >= start_time and v.end_time < end_time and abs(v.end_position - v.start_position) > threshold:
        movement_count += 1
    return movement_count

  def get_movement_average_velocity(self, digit, start_time=None, end_time=None, threshold=0):
    """
    Calculates the average velocity of any movement within the specified time period
    :param digit: index or thumb, the digit to calculate the data for
    :param start_time: the start time of the analysis period
    :param end_time: the end time of the analysis period
    :param threshold: the minimum value change for the movement to be considered valid
    :return: the average velocity
    """
    if digit not in self.velocity_change_data:
      return
    if start_time is None:
      start_time = 0
    if end_time is None:
      end_time = self.velocity_change_data[digit][-1].end_time
    if start_time > self.velocity_change_data[digit][-1].end_time:
      return 0
    total_velocity = 0
    count = 0
    for v in self.velocity_change_data[digit]:
      if v.start_time >= start_time and v.end_time < end_time and abs(v.end_position - v.start_position) > threshold:
        count += 1
        total_velocity += abs(v.end_position - v.start_position)/(v.end_time - v.start_time)
    return 0 if count == 0 else total_velocity/count

  def get_movement_average_magnitude(self, digit, start_time=None, end_time=None, threshold=0):
    """
    Calculates the average magnitude of the movement within the specified time period
    :param digit: index or thumb, the digit to analyse
    :param start_time: the start time of the analysis period
    :param end_time: the end time of the analysis period
    :param threshold: the minimum movement to be considered valid
    :return: the total distance, the number of movements, and the average movement
    """
    if digit not in self.velocity_change_data:
      return
    if start_time is None:
      start_time = 0
    if end_time is None:
      end_time = self.velocity_change_data[digit][-1].end_time
    if start_time > self.velocity_change_data[digit][-1].end_time:
      return 0
    total_distance = 0
    count = 0
    for v in self.velocity_change_data[digit]:
      if v.start_time >= start_time and v.end_time < end_time and abs(v.end_position - v.start_position) > threshold:
        count += 1
        total_distance += abs(v.end_position - v.start_position)
    return (0,0,0) if count == 0 else (total_distance, count, total_distance/count)

  def calc_idle_time(self, digit, start_time=None, end_time=None, threshold=0):
    """
    Calculates the idle time
    :param digit: index or thumb, the digit to analyse
    :param start_time: the start time of the analysis period
    :param end_time: the end time of the analysis period
    :param threshold: the minimum movement to be considered valid
    :return: The longest idle time (seconds) and total idle time (seconds)
    """
    if digit not in self.velocity_change_data:
      return
    if start_time is None:
      start_time = 0
    if end_time is None:
      end_time = self.velocity_change_data[digit][-1].end_time
    blah = self.velocity_change_data[digit][-1].end_time
    if start_time > self.velocity_change_data[digit][-1].end_time:
      return 0, 0
    total_idle_time = 0
    longest_idle_time = 0
    tmp_idle_time = 0
    for v in self.velocity_change_data[digit]:
      if v.start_time >= start_time and v.end_time <= end_time:
        if abs(v.end_position - v.start_position) > threshold and tmp_idle_time > 0:
          if tmp_idle_time > longest_idle_time:
            longest_idle_time = tmp_idle_time
          total_idle_time += tmp_idle_time
          tmp_idle_time = 0
        else:
          tmp_idle_time += v.end_time - v.start_time
    return longest_idle_time, total_idle_time

  def process_file(self):
    """
    Reads the file into the data array
    :return:
    """
    tmp_time = 1/self.freq
    with open(self.file_name, 'r') as f:
      csv_reader = csv.reader(f, delimiter=',', quotechar='"')
      for line in csv_reader:
        self.data.append(PositionData(tmp_time, line[self.index_col], line[self.thumb_col]))
        tmp_time += 1/self.freq
    self.generate_velocity_change_data()

  def print_velocity_data(self, filename=None):
    """
    Prints the velocity data or writes it to the supplied filename
    :param filename: Optional, the file to write the data to
    :return:
    """
    if not filename is None:
      f = open(filename, 'w')
    for k,v in self.velocity_change_data.items():
      if filename is None:
        print('%s Velocity Data' % k)
        print('Start Time,Time,Distance')
      else:
        f.write('%s Velocity Data\n' % k)
        f.write('Start Time,Time,Distance\n')
      for a in v:
        if filename is None:
          print(a)
        else:
          f.write('%s\n' % a)
    if not filename is None:
      f.close()

  def get_average(self, digit, start_point=None, end_point=None):
    """
    Generates an average position for the sample time
    :param digit: index or thumb, the digit to analyse
    :param start_point: The start time of the sample period
    :param end_point: the end time of the sample period
    :return:
    """
    if start_point is None:
      start_point = 0
    if end_point is None:
      end_point = len(self.data)

    total = 0
    for d in self.data[start_point:end_point]:
      total += d.thumb_pos if digit == 'thumb' else d.index_pos
    return total/len(self.data[start_point:end_point]) if len(self.data[start_point:end_point]) > 0 else None

  def generate_velocity_change_data(self):
    """
    Generates the velocity change data of the sample
    :return:
    """
    for d in ['index', 'thumb']:
      for i in range(1, len(self.data)):
        if self.data[i-1].get_position_data(d) > self.data[i].get_position_data(d):
          direction = -1
          break
        elif self.data[i-1].get_position_data(d) < self.data[i].get_position_data(d):
          direction = 1
          break
      data_point = self.data[0]
      index = 1
      while data_point is not None:
        v = VelocityChanges(direction, data_point.timestamp, data_point.get_position_data(d))
        direction, data_point, index = self.find_direction_change( index, v, d)
        if data_point is None:
          v.complete(self.data[-1].timestamp, self.data[-1].get_position_data(d))
        self.velocity_change_data[d].append(v)

  def find_direction_change(self, index, velocity_data, digit):
    """
    Finds a change in direction from position data
    :param index: The start point to determine the direction change from
    :param velocity_data: The existing velocity data object for the current movement
    :param digit: index or thumb to analyse
    :return:
    """
    for i in range(index, len(self.data)):
      new_direction = velocity_data.check_value(self.data[i].get_position_data(digit))
      if new_direction != velocity_data.direction and new_direction != 0:
        velocity_data.complete(self.data[i-1].timestamp, self.data[i-1].get_position_data(digit))
        return new_direction, self.data[i-1], i
    return None, None, None

  def generate_velocity_change_plot(self, filename):
    """
    Generates a matplotlib of the velocity data
    :param filename: The file to save the plot into
    :return:
    """
    for d in ['index', 'thumb']:
      plot_filename = filename % d
      position_change = []
      time_data = []
      for data in self.velocity_change_data[d]:
        position_change.append(data.end_position - data.start_position)
        time_data.append(data.start_time)
      fig = plt.figure(figsize=(20, 10))
      sp = fig.add_subplot(1, 1, 1)
      sp.set_xlim(0, self.data[-1].timestamp)
      sp.set_xlabel('Time (s)')
      sp.set_ylabel('Arbitary Units')
      sp.plot(time_data, position_change)
      fig.savefig(plot_filename)
      plt.close()

  def get_plot_data(self, digit, start_time=0, end_time=0):
    """
    Get the Time and Position aray for a given digit
    :param digit: index or thumb, the digit to get the data for
    :return:
    """
    time_array = []
    position_array = []
    if end_time == 0:
      end_time = self.data[-1].timestamp
    for d in self.data:
      if start_time <= d.timestamp <= end_time:
        time_array.append(d.timestamp)
        position_array.append(d.get_position_data(digit))
      if d.timestamp > end_time:
        break
    return time_array, position_array

class PositionData:
  """
  Holds the position data
  """
  def __init__(self, timestamp, index_pos, thumb_pos):
    self.timestamp = timestamp
    self.thumb_pos = float(thumb_pos)
    self.index_pos = float(index_pos)

  def get_position_data(self, digit):
    """
    Gets the position data for a given digit
    :param digit: thumb or index, the digit to get the data for
    :return:
    """
    return self.thumb_pos if digit == 'thumb' else self.index_pos

class CalibrationData:
  """
  The Calibration data of the subject
  """
  def __init__(self, sample_data, digit):
    self.sample_data = sample_data
    self.digit = digit
    self.sample_points = []
    self.sample_set = []

  def configure(self, frequency, offset, sample_size, step_size, sample_count):
    """
    Generates the sample points from the parameters in the config file
    :param frequency: the frequency/sample rate of the data file
    :param offset: The initial offset to take the first sample from
    :param sample_size: the time (seconds) of the sample
    :param step_size: the time difference between the samples
    :param sample_count: the number of samples to take.
    :return:
    """
    for i in range(sample_count):
      self.sample_points.append([frequency * (offset + step_size * i), frequency * (offset + step_size * i + sample_size)])

  def generate_sample_set(self):
    for sp in self.sample_points:
      if self.sample_data is not None:
        avg = self.sample_data.get_average(self.digit, sp[0], sp[1])
        if avg is not None:
          self.sample_set.append(avg)

  def get_average(self):
    return numpy.mean(self.sample_set)

  def get_min(self):
    return numpy.amin(self.sample_set)

  def get_max(self):
    return numpy.amax(self.sample_set)

  def add_to_plot(self, plot):
    for i in range(len(self.sample_set)):
      plot.plot(i+1, self.sample_set[i], 'ko')

  def __str__(self):
    return 'max = %0.2f, min = %0.2f, average = %0.2f' % (self.get_max(), self.get_min(), self.get_average())

class VelocityChanges:
  def __init__(self, direction, start_time, start_position):
    self.direction = direction
    self.start_time = start_time
    self.start_position = start_position
    self.end_time = None
    self.end_position = None
    self.last_position = start_position

  def check_value(self, check_position):
    if check_position == self.last_position:
      return 0
    if check_position < self.last_position:
      self.last_position = check_position
      return -1
    if check_position > self.last_position:
      self.last_position = check_position
      return 1

  def complete(self, time_value, position):
    self.end_position = position
    self.end_time = time_value

  def __str__(self):
    return '%0.2f,%0.2f,%0.2f' % (self.start_time, self.end_time - self.start_time, self.end_position - self.start_position)
    # return 'Start Time = %0.2f, Time = %0.2f, Distance = %0.2f' % (self.start_time, self.end_time - self.start_time, self.end_position - self.start_position)

def generate_figure(ylabel):
  fig = plt.figure(figsize=(10, 10))
  index_sp = fig.add_subplot(2, 1, 1)
  index_sp.set_title('Index')
  thumb_sp = fig.add_subplot(2, 1, 2)
  thumb_sp.set_title('Thumb')
  for sp in [index_sp, thumb_sp]:
    sp.set_xlim(5, 25)
    sp.set_xlabel('Patient')
    sp.set_xticks([10, 20])
    sp.set_xticklabels(['Control', 'Stroke'])
    sp.set_ylabel(ylabel)
  return fig, index_sp, thumb_sp


def main():
  config = ConfigParser()
  config.read('config.ini')
  defaults = {}

  for c in config.options('Defaults'):
    if c == 'patients' or c == 'digits':
      defaults.update({ c : [x.strip() for x in config.get('Defaults', c).split(',')] })
    elif c.endswith('_calibration'):
      defaults.update({ c : [int(x) for x in config.get('Defaults', c).split(',')] })
    elif c.find('_thresholds_') != -1:
      defaults.update({ c : [float(x) for x in config.get('Defaults', c).split(',')] })

    else:
      try:
        v = config.getint('Defaults', c)
      except ValueError:
        v = config.get('Defaults', c)
      defaults.update({ c : v })
  # print(defaults)

  min_threshold = float(defaults['movement_threshold'])

  percentile_count = int(defaults['percentile_count'])

  """
  percentile_filename = 'Histogram Position Data Raw.csv'
  percentile_filename_percentage = 'Histogram Position Data Percentage.csv'
  f1 = open(percentile_filename, 'w')
  f2 = open(percentile_filename_percentage, 'w')
  percentile_file_raw = csv.writer(f1, delimiter=',', quotechar='"')
  percentile_file_percentage = csv.writer(f2, delimiter=',', quotechar='"')
  output_array = ['Subject Name', 'Digit', '< 0']
  for i in range(percentile_count):
    output_array.append('%d - %d' % (i*percentile_count, (i+1)*percentile_count))
  output_array.append('> 100')
  percentile_file_raw.writerow(output_array)
  percentile_file_percentage.writerow(output_array)
  """

  subjects = []
  for p in defaults['patients']:
    settings = defaults.copy()
    if p in config.sections():
      for option in config.options(p):
        if option == 'digits':
          settings[option] = [x.strip() for x in config.get(p, option).split(',')]
        elif option.endswith('_calibration'):
          settings[option] = [int(x) for x in config.get(p, option).split(',')]
        else:
          try:
            settings[option] = config.getint(p, option)
          except ValueError:
            settings[option] = config.get(p, option)
    s = Subject(p, settings)
    for f in ['calibration1', 'trial', 'calibration2']:
      if '%s_frequency' % f in settings:
        frequency = settings['%s_frequency' % f]
      else:
        frequency = settings['frequency']
      if '%s_index_col' % f in settings:
        index_col = settings['%s_frequency' % f]
      else:
        index_col = settings['index_col']
      if '%s_thumb_col' % f in settings:
        thumb_col = settings['%s_thumb_col' % f]
      else:
        thumb_col = settings['thumb_col']
      s.read_sample_file(f, index_col, thumb_col, frequency)
    for f in ['calibration1', 'calibration2']:
      for d in settings['digits']:
        for t in ['min', 'max']:
          settings_lookup = '%s_%s_%s_calibration' % (f, d, t)
          if settings_lookup in settings:
            data = settings[settings_lookup]
          else:
            settings_lookup = '%s_%s_calibration' % (d, t)
            data = settings[settings_lookup]
          if '%s_frequency' % f in settings:
            frequency = settings['%s_frequency' % f]
          else:
            frequency = settings['frequency']
          s.generate_calibration_data(f, d, t, frequency, data[0], data[1], data[2], data[3])
    s.generate_calibration_plot()
    s.calc_calibration_values()

    print('%s - Producing Calibration Plots for %s' % (datetime.now().strftime('%H:%M:%S'), s.folder_name))
    for f in ['calibration1', 'trial', 'calibration2']:
      for d in ['index', 'thumb']:
        s.produce_position_plot(f, d)
    print('%s - Producing Velocity Change Plots for %s' % (datetime.now().strftime('%H:%M:%S'), s.folder_name))
    s.generate_velocity_change_plots()

    movement_count_thresholds = {'index': settings['movement_count_thresholds_index'], 'thumb': settings['movement_count_thresholds_thumb']}
    velocity_thresholds = { 'index' : settings['velocity_thresholds_index'], 'thumb' : settings['velocity_thresholds_thumb']}

    print('%s - Producing Movement and Velocity Histogram Data for %s' % (datetime.now().strftime('%H:%M:%S'), s.folder_name))
    s.produce_histograms(movement_count_thresholds, velocity_thresholds, settings['segment_time'], min_threshold)
    s.get_average_movement_amplitude(min_threshold)

    print('%s - Getting Idle Times for %s' % (datetime.now().strftime('%H:%M:%S'), s.folder_name))
    s.get_idle_times(min_threshold)

    print('%s - Generating Position Histogram Data for %s' % (datetime.now().strftime('%H:%M:%S'), s.folder_name))
    s.calc_histogram_plot(percentile_count)

    print('%s - Writing Summary File for %s' % (datetime.now().strftime('%H:%M:%S'), s.folder_name))
    s.write_summary(settings)
    """
    print('%s - %s' % (s.folder_name, i))
    print('Average Amplitude: %0.2f' % average_amplitude)
    print(movement_groupings)
    print(velocity_groupings)
    print('longest idle time: %0.2f' % longest_idle_time)
    print('total_idle_time: %0.2f%%' % (total_idle_time / end_time * 100))



    
    
    s.write_histogram_to_file(s.folder_name, percentile_file_raw, percentile_file_percentage)
    s.generate_velocity_change_plots()
    end_time = s.data_set.data['trial'].data[-1].timestamp
    # movement_count_thresholds = { 'index' : [1.5, 1.7, 2.7], 'thumb' : [1.65, 3, 4.5] }
    # velocity_thresholds = { 'index' : [7.1, 12.5, 21.5], 'thumb' : [6.3, 12, 20.8] }
    
      for t in range(1800, math.ceil(end_time), 1800):
        sample_start_time = (t - 1800)
        sample_end_time = t
        longest_idle_time, total_idle_time = s.data_set.data['trial'].calc_idle_time(i, sample_start_time, sample_end_time, min_threshold)
        print('idle data for %s' % i)
        print('longest idle time: %0.2f' % longest_idle_time)
        print('total idle time: %0.2f\n' % total_idle_time)
    """
    subjects.append(s)
  sp = plt.subplot(1, 1, 1)
  sp.set_xlim(5, 25)
  sp.set_xlabel('Calibration')
  sp.set_xticks([10, 20])
  sp.set_xticklabels(['calibration1', 'calibration2'])
  sp.set_ylabel('Arbitary Units')

  for s in subjects:
    for d in s.digit_list:
      s.add_calibration_data_to_plot(sp, d)

  plt.savefig('graphs/summary/full_calibration.png')
  plt.close()

  x_histo = ['< 0%']
  size = 100/defaults['percentile_count']
  for i in range(defaults['percentile_count']):
    x_histo.append('%d%% - %d%%' % (i*size, (i+1)*size))
  x_histo.append('> 100%%')
  x_points = [x for x in range(1, defaults['percentile_count']+3)]

  position_data = {}
  for i in ['Control', 'Stroke']:
    position_data.update({i:{}})
    for d in defaults['digits']:
      position_data[i].update({d:[]})

  with open('summary/log.csv', 'w') as f:
    csv_writer = csv.writer(f, quotechar='"', delimiter=',')
    csv_writer.writerow(
      ['subject', 'group', 'test_date', 'age', 'sex', 'stroke_date', 'stroke_type', 'side_tested', 'mas_score_7',
       'mas_score_8', 'dominiant_hand'])
    for s in subjects:
      s.write_log_to_csv(csv_writer)

  header_prefix = ['subject', 'group', 'age', 'sex']

  header_raw = header_prefix.copy()
  header_percent = header_prefix.copy()
  header_raw.append('total_time_points')
  header_raw.append('lt_0_pct')
  header_percent.append('lt_0_pct')
  for i in range(defaults['percentile_count']):
    header_raw.append('ge_%d_pct_to_lt_%d_pct' % (i * defaults['percentile_count'], (i + 1) * defaults['percentile_count']))
    header_percent.append('ge_%d_pct_to_lt_%d_pct' % (i * defaults['percentile_count'], (i + 1) * defaults['percentile_count']))
  header_raw.append('ge_100_pct')
  header_percent.append('ge_100_pct')

  empty_data = ['NA'] * (defaults['percentile_count']+2)

  for d in defaults['digits']:
    raw_file_handler = open('summary/instantenous_position_raw_%s.csv' % d, 'w')
    percentage_file_handler = open('summary/instantenous_position_percentage_%s.csv' % d, 'w')
    raw_csv_file = csv.writer(raw_file_handler, quotechar='"', delimiter=',')
    percentage_csv_file = csv.writer(percentage_file_handler, quotechar='"', delimiter=',')
    raw_csv_file.writerow(header_raw)
    percentage_csv_file.writerow(header_percent)
    for s in subjects:
      if d in s.position_histogram:
        s.position_histogram[d].write_to_file(s.generate_output_file_prefix() + [len(s.data_set.data['trial'].data)], raw_csv_file)
        s.position_histogram[d].write_to_file(s.generate_output_file_prefix(), percentage_csv_file, False)
      else:
        raw_csv_file.writerow(s.generate_output_file_prefix() + empty_data + ['NA'])
        percentage_csv_file.writerow(s.generate_output_file_prefix() + empty_data)
    raw_file_handler.close()
    percentage_file_handler.close()

  for d in defaults['digits']:
    amplitude_file = open('summary/amplitude_%s.csv' % d, 'w')
    amplitude_csv = csv.writer(amplitude_file, quotechar='"', delimiter=',')
    amplitude_csv.writerow(header_prefix + ['total_amplitude', 'total_movements', 'average_amplitude', 'pct_rom'])
    for s in subjects:
      if d in s.average_amplitude:
        amplitude_csv.writerow(s.generate_output_file_prefix() + [s.total_amplitude[d], s.movement_count[d], s.average_amplitude[d], (s.average_amplitude[d]/s.ROM[d])])
      else:
        amplitude_csv.writerow(s.generate_output_file_prefix() + ['NA'] * 4)
    amplitude_file.close()

  for d in defaults['digits']:
    velocity_file = open('summary/velocity_%s.csv' % d, 'w')
    velocity_csv = csv.writer(velocity_file, quotechar='"', delimiter=',')
    header = header_prefix.copy()
    header += ['total_movements', 'total_time_min', 'average_velocity']
    header.append(('lt_%0.2f' % defaults['velocity_thresholds_index'][0]).replace('.', 'dot'))
    for i in range(len(defaults['velocity_thresholds_%s' % d])-1):
      header.append(('ge_%0.2f_to_lt_%0.2f' % (defaults['velocity_thresholds_%s' % d][i], defaults['velocity_thresholds_%s' % d][i+1])).replace('.', 'dot'))
    header.append(('ge_%0.2f' % defaults['velocity_thresholds_%s' % d][-1]).replace('.', 'dot'))
    velocity_csv.writerow(header)
    for s in subjects:
      if d in s.digit_list:
        velocity_csv.writerow(s.generate_output_file_prefix() + [s.movement_count[d], s.data_set.data['trial'].data[-1].timestamp/60, s.average_velocity[d]/s.ROM[d]] + s.velocity_groupings[d])
      else:
        velocity_csv.writerow(s.generate_output_file_prefix() + ['NA'] * (3 + len(defaults['velocity_thresholds_%s' % d]) + 1))
    velocity_file.close()

  for d in defaults['digits']:
    movement_rate_file = open('summary/movement_rate_%s.csv' % d, 'w')
    movement_rate_csv = csv.writer(movement_rate_file, quotechar='"', delimiter=',')
    header = header_prefix.copy()
    header += ['total_movements', 'total_time_min', 'movements_per_sec']
    header.append(('lt_%0.2f' % defaults['movement_count_thresholds_%s' % d][0]).replace('.', 'dot'))
    for i in range(len(defaults['movement_count_thresholds_%s' % d])-1):
      header.append(('ge_%0.2f_to_lt_%0.2f' % (defaults['movement_count_thresholds_%s' % d][i], defaults['movement_count_thresholds_%s' % d][i+1])).replace('.', 'dot'))
    header.append(('ge_%0.2f' % defaults['movement_count_thresholds_%s' % d][-1]).replace('.', 'dot'))
    movement_rate_csv.writerow(header)
    for s in subjects:
      if d in s.digit_list:
        movement_rate_csv.writerow(s.generate_output_file_prefix() + [s.movement_count[d], s.data_set.data['trial'].data[-1].timestamp/60, s.average_movement[d]] + s.movement_groupings[d])
      else:
        movement_rate_csv.writerow(s.generate_output_file_prefix() + ['NA'] * (3 + len(defaults['movement_count_thresholds_%s' % d]) + 1))
    movement_rate_file.close()

  for d in defaults['digits']:
    idle_time_file = open('summary/idle_time_%s.csv' % d, 'w')
    idle_time_csv = csv.writer(idle_time_file, quotechar='"', delimiter=',')
    header = header_prefix.copy()
    header += ['total_idle_time_min', 'total_time_min', 'pct_idle_time', 'longest_idle_time_sec']
    idle_time_csv.writerow(header)
    for s in subjects:
      if d in s.digit_list:
        idle_time_csv.writerow(s.generate_output_file_prefix() + [s.total_idle_time[d]/60, s.data_set.data['trial'].data[-1].timestamp/60, s.total_idle_time[d]/s.data_set.data['trial'].data[-1].timestamp * 100, s.longest_idle_time[d]])
      else:
        idle_time_csv.writerow(s.generate_output_file_prefix() + ['NA'] * 4)
    idle_time_file.close()

  for d in defaults['digits']:
    calibration_file = open('summary/calibration_%s.csv' % d, 'w')
    calibration_csv = csv.writer(calibration_file, quotechar='"', delimiter=',')
    header = header_prefix.copy()
    header += ['calibration1_min', 'calibration1_max', 'calibration2_min', 'calibration2_max', 'difference_min', 'difference_max', 'average_min', 'average_max', 'rom']
    calibration_csv.writerow(header)
    for s in subjects:
      if d in s.digit_list:
        output = []
        for c in ['calibration1', 'calibration2']:
          for m in ['min', 'max']:
            if (c in s.calibration_data and m in s.calibration_data[c][d]) and s.calibration_data[c][d][m] is not None:
              output.append(s.calibration_data[c][d][m])
            else:
              output.append('NA')
        if s.calibration_data['calibration1'][d]['min'] is not None and s.calibration_data['calibration2'][d]['min'] is not None:
          for m in ['min', 'max']:
            output.append(s.calibration_data['calibration1'][d][m] - s.calibration_data['calibration2'][d][m])
          for m in ['min', 'max']:
            output.append(numpy.mean([s.calibration_data['calibration1'][d][m], s.calibration_data['calibration2'][d][m]]))
        else:
          output += ['NA'] * 4
        if output[6] == 'NA':
          if s.calibration_data['calibration1'][d]['min'] is None:
            output.append(s.calibration_data['calibration2'][d]['max'] - s.calibration_data['calibration2'][d]['min'])
          else:
            output.append(s.calibration_data['calibration1'][d]['max'] - s.calibration_data['calibration1'][d]['min'])
        else:
          output.append(output[7] - output[6])
        calibration_csv.writerow(s.generate_output_file_prefix() + output)
    calibration_file.close()


  for s in subjects:
    fig = plt.figure(figsize=(10, 5*len(s.digit_list)))
    for i in range(len(s.digit_list)):
      axes = fig.add_subplot(len(s.digit_list), 1, i+1)
      axes.set_title(s.digit_list[i].capitalize())
      axes.set_ylabel('% Time')
      axes.set_xlabel('% of Flexion')
      axes.set_xticks(x_points)
      axes.set_xticklabels(x_histo)
      tmp_array = s.position_histogram[s.digit_list[i]].get_percentages_array()
      axes.bar(x_points, tmp_array)
      if s.folder_name.startswith('CON'):
        patient_type = 'Control'
      else:
        patient_type = 'Stroke'
      position_data[patient_type][s.digit_list[i]].append(tmp_array)

      # axes[i].bar([ x for x in range(1, defaults['percentile_count']+1)], s.position_histogram[s.digit_list[i]].get_percentages_array())
      # axes[i].hist(s.position_histogram[s.digit_list[i]].get_percentages_array(), bins=defaults['percentile_count'] + 2)
    fig.tight_layout()
    fig.savefig('graphs/%s_position_histogram.png' % s.folder_name)
    plt.close(fig)

  for p in ['Control', 'Stroke']:
    fig = plt.figure(figsize=(10, 10))
    for i in range(len(defaults['digits'])):
      axes = fig.add_subplot(2, 1, i+1)
      axes.set_title(defaults['digits'][i].capitalize())
      axes.set_ylabel('% Time')
      axes.set_xlabel('% of Flexion')
      axes.set_xticks(x_points)
      axes.set_xticklabels(x_histo)
      axes.bar(x_points, np.average(position_data[p][defaults['digits'][i]], 0))
      for d in position_data[p][defaults['digits'][i]]:
        axes.plot(x_points, d, linewidth=1)
    fig.tight_layout()
    fig.savefig('graphs/summary/%s_average_position_histogram.png' % p)
    plt.close(fig)

  fig, index_sp, thumb_sp = generate_figure('Average Movements/Sec')
  for s in subjects:
    for d in s.digit_list:
      if s.folder_name.startswith('CON'):
        x_coord = 10
      else:
        x_coord = 20
      if d == 'thumb':
        plot = thumb_sp
      else:
        plot = index_sp
      plot.plot(x_coord, s.average_movement[d], 'ks', markersize=3)
  fig.tight_layout()
  fig.savefig('graphs/summary/average_movement.png')
  plt.close(fig)

  fig, index_sp, thumb_sp = generate_figure('Average Velocity (% ROM/sec)')
  for s in subjects:
    for d in s.digit_list:
      if s.folder_name.startswith('CON'):
        x_coord = 10
      else:
        x_coord = 20
      if d == 'thumb':
        plot = thumb_sp
      else:
        plot = index_sp
      plot.plot(x_coord, s.average_velocity[d]/s.ROM[d], 'ks', markersize=3)
  fig.tight_layout()
  fig.savefig('graphs/summary/average_velocity.png')
  plt.close(fig)

  fig, index_sp, thumb_sp = generate_figure('Average Amplitude (%ROM)')
  for s in subjects:
    for d in s.digit_list:
      if s.folder_name.startswith('CON'):
        x_coord = 10
      else:
        x_coord = 20
      if d == 'thumb':
        plot = thumb_sp
      else:
        plot = index_sp
      plot.plot(x_coord, s.average_amplitude[d]/s.ROM[d], 'ks', markersize=3)
  fig.tight_layout()
  fig.savefig('graphs/summary/average_amplitude.png')
  plt.close(fig)

  fig, index_sp, thumb_sp = generate_figure('% Time')
  for s in subjects:
    for d in s.digit_list:
      if s.folder_name.startswith('CON'):
        x_coord = 10
      else:
        x_coord = 20
      if d == 'thumb':
        plot = thumb_sp
      else:
        plot = index_sp
      plot.plot(x_coord, s.total_idle_time[d]/s.data_set.data['trial'].data[-1].timestamp*100, 'ks', markersize=3)
  fig.tight_layout()
  fig.savefig('graphs/summary/total_idle_time.png')
  plt.close(fig)

  fig, index_sp, thumb_sp = generate_figure('Time (s)')
  for s in subjects:
    for d in s.digit_list:
      if s.folder_name.startswith('CON'):
        x_coord = 10
      else:
        x_coord = 20
      if d == 'thumb':
        plot = thumb_sp
      else:
        plot = index_sp
      plot.plot(x_coord, s.longest_idle_time[d], 'ks', markersize=3)
  fig.tight_layout()
  fig.savefig('graphs/summary/longest_idle_time.png')
  plt.close(fig)


if __name__ == '__main__':
  main()