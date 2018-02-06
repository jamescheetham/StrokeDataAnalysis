import csv, os, sys, numpy
from optparse import OptionParser
import matplotlib.pyplot as plt
import matplotlib as mpl

class VelocityData:
  def __init__(self, start_time, start_position, direction, freq=1000):
    self.start_time = start_time
    self.start_position = start_position
    self.direction = direction
    self.freq = freq
    self.last_position = start_position
    self.last_timestamp = start_time
    self.end_position = 0
    self.end_time = 0

  def get_distance(self):
    return self.end_position - self.start_position

  def get_time(self):
    return (self.end_time - self.start_time)/self.freq

  def get_velocity(self):
    return self.get_distance()/self.get_time()

  def check_data(self, timestamp, position):
    if self.direction == 1 and self.last_position <= position:
      self.last_position = position
      self.last_timestamp = timestamp
      return True
    elif self.direction == -1 and self.last_position >= position:
      self.last_position = position
      self.last_timestamp = timestamp
      return True
    else:
      self.end_position = self.last_position
      self.end_time = self.last_timestamp
      return False
    
  def enter_final_value(self, timestamp, position):
    self.end_position = position
    self.end_time = timestamp
    
  def __str__(self):
    return "d = %0.2f, t = %0.2f, initial time = %d" % (self.end_position - self.start_position, (self.end_time - self.start_time)/self.freq, self.start_time)

class ColumnData:
  def __init__(self, col_data_file, delimiter=':'):
    self.columndata = []
    with open(col_data_file, 'r') as f:
      for line in f:
        self.columndata.append([x.strip() for x in line.split(delimiter)])
  
  def get_col_name(self, pos):
    return self.columndata[pos][1]
  
  def get_col_type(self, pos):
    return self.columndata[pos][2]
  
  def get_col_num(self, pos):
    return int(self.columndata[pos][0])
  
  def get_col_data(self, pos):
    return self.columndata[pos]
  
  def get_col_count(self):
    return len(self.columndata)

class FileData:
  def __init__(self, filename, col_data):
    self.data = []
    with open(filename, 'r') as f:
      csv_data = csv.reader(f, delimiter=',', quotechar='"')
      for linedata in csv_data:
        self.data.append(FileDataLine(linedata, col_data))
        
  def get_data(self, index):
    return self.data[index]
        
  def get_data_count(self):
    return len(self.data)
        
  def get_max(self, col_name):
    max_value = None
    for d in self.data:
      if max_value is None or d.get_value(col_name) > max_value:
        max_value = d.get_value(col_name)
    return max_value
  
  def get_min(self, col_name):
    min_value = None
    for d in self.data:
      if min_value is None or d.get_value(col_name) < min_value:
        min_value = d.get_value(col_name)
    return min_value
  
  def get_average(self, col_name, start_point=None, end_point=None):
    if start_point is None:
      start_point = 0
    if end_point is None:
      end_point = len(self.data)
    total = 0
    for d in self.data[start_point:end_point]:
      total += d.get_value(col_name)
    return total/(end_point - start_point)
  
  def get_std_dev(self, col_name):
    tmp_data = []
    for d in self.data:
      tmp_data.append(d.get_value(col_name))
    arr = numpy.array(tmp_data)
    return numpy.std(arr, ddof=1)
  
  def get_max_step(self, col_name):
    max_step = None
    for i in range(1, len(self.data)):
      prior = self.data[i-1]
      current = self.data[i]
      diff = abs(prior.get_value(col_name) - current.get_value(col_name))
      if max_step is None or max_step < diff:
        max_step = diff
    return max_step
  
  def get_max_step_percent(self, col_name):
    max_step_percent = None
    for i in range(1, len(self.data)):
      prior = self.data[i-1]
      current = self.data[i]
      diff = abs(prior.get_value(col_name) - current.get_value(col_name))/prior.get_value(col_name)
      if max_step_percent is None or max_step_percent < diff:
        max_step_percent = diff
    return max_step_percent * 100
  
  def construct_histogram_data(self, col_name, buckets=None, bucket_count=None):
    min_value = self.get_min(col_name)
    max_value = self.get_max(col_name)

    if buckets is None and bucket_count is None:
      bucket_count = 10
    if buckets is None:
      buckets = []
      step = (max_value - min_value)/(bucket_count)
      floor = min_value
      ceiling = min_value + step
      for i in range(bucket_count):
        buckets.append(HistogramData(floor, ceiling))
        floor += step
        ceiling = floor + step
    for d in self.data:
      for b in buckets:
        b.add_value(d.get_value(col_name))
    return buckets
  
  def create_average_set(self, col_name, sample_size):
    average_set = []
    for i in range(0, len(self.data) - sample_size):
      total = 0
      for d in self.data[i:i+sample_size]:
        total += d.get_value(col_name)
      average_set.append(total/sample_size)
    return average_set
  
class HistogramData:
  def __init__(self, floor, ceiling):
    self.floor = floor
    self.ceiling = ceiling
    self.count = 0
    
  def add_value(self, value):
    if value < self.ceiling and value >= self.floor:
      self.count += 1
      
  def __str__(self):
    return '%0.2f - %0.2f: %d' % (self.floor, self.ceiling, self.count)

class FileDataLine:
  def __init__(self, linedata, col_data):
    self.data = {}
    try:
      for i in range(col_data.get_col_count()):
        col_num = col_data.get_col_num(i)
        if col_data.get_col_type(i) == 'int':
          tmp_data = int(linedata[col_num - 1])
        elif col_data.get_col_type(i) == 'float':
          tmp_data = float(linedata[col_num - 1])
        else:
          tmp_data = linedata[col_num - 1]
        self.data.update({ col_data.get_col_name(i) : tmp_data })
    except IndexError:
      print('Index Error processing FileDataLine')
      print('Line Data = %s' % linedata)
      print('Index = %d' % i)
      sys.exit(1)

  
  def get_value(self, col_name):
    try:
      return self.data[col_name]
    except KeyError:
      print('KeyError')
      print('col_name = %s' % col_name)
      print('self.data = %s' % self.data)
      sys.exit(1)
      
  def __str__(self):
    output = ""
    for k, v in self.data.items():
      output += "k = %s, v = %0.2f\n" % (k, v)
    return output
      
class CalibrationData:
  def __init__(self, file_data, column, time_col):
    self.file_data = file_data
    self.column = column
    self.time_col = time_col
    
  def configure(self, offset, sample_size, frequency, step_size, sample_count):
    self.sample_points = []
    for i in range(sample_count):
      self.sample_points.append([frequency*(offset+step_size*i), frequency*(offset+step_size*i+sample_size)])
      
  def generate_sample_set(self):
    self.sample_data = []
    for sp in self.sample_points:
      #print(self.file_data.data[sp[0]].get_value('timestamp'))
      self.sample_data.append(self.file_data.get_average(self.column, sp[0], sp[1]))
  
  def get_average(self):
    return numpy.mean(self.sample_data)
  
  def get_min(self):
    return numpy.amin(self.sample_data)
  
  def get_max(self):
    return numpy.amax(self.sample_data)
        
def main():
  parser = OptionParser()
  parser.add_option('-c', '--config-file', type='string', action='store', dest='config_file')
  parser.add_option('-f', '--input-file', type='string', action='store', dest='input_file')
  parser.add_option('-C', '--column-name', type='string', action='store', dest='column_name')
  
  (opt, arg) = parser.parse_args()
  
  if opt.config_file is None:
    parser.error('Please provide a config file')
    sys.exit(1)
  if opt.input_file is None:
    parser.error('Please provide an input file')
    sys.exit(1)
  if not os.path.isfile(opt.config_file):
    parser.error('The config file %s does not exist or is not readable' % (opt.config_file))
    sys.exit(1)
  if not os.path.isfile(opt.input_file):
    parser.error('The input file %s does not exist or is not readable' % (opt.input_file))
    
  column_data = ColumnData(opt.config_file)
  
  if opt.column_name is None:
    for i in range(column_data.get_col_count()):
      if column_data.get_col_type(i) in ['float', 'int']:
        opt.column_name = column_data.get_col_name(i)
        break
    
  fd = FileData(opt.input_file, column_data)
  
  cd = CalibrationData(fd, 'sensor1', 'timestamp')
  cd.configure(6, 2, 10, 30, 5)
  cd.generate_sample_set()
  thumb_ext = cd.get_min()
  #print('Thumb Extension')
  #print('min = %0.2f' % cd.get_min())
  #print('max = %0.2f' % cd.get_max())
  #print('avg = %0.2f' % cd.get_average())
  #print('data set = %s\n' % (', '.join(str(v) for v in cd.sample_data)))

  cd.configure(26, 2, 10, 30, 5)
  cd.generate_sample_set()
  thumb_flex = cd.get_max()
  #print('Thumb Flexion')
  #print('min = %0.2f' % cd.get_min())
  #print('max = %0.2f' % cd.get_max())
  #print('avg = %0.2f' % cd.get_average())
  #print('data set = %s\n' % (', '.join(str(v) for v in cd.sample_data)))
  
  cd = CalibrationData(fd, 'sensor2', 'timestamp')
  cd.configure(6, 2, 10, 30, 5)
  cd.generate_sample_set()
  if_ext = cd.get_min()
  #print('Index Finger Extension')
  #print('min = %0.2f' % cd.get_min())
  #print('max = %0.2f' % cd.get_max())
  #print('avg = %0.2f' % cd.get_average())

  cd.configure(16, 2, 10, 30, 5)
  cd.generate_sample_set()
  if_flex = cd.get_max()
  
  print('Index Finger min, max = %0.2f, %0.2f' % (if_ext, if_flex))
  print('Diff = %0.2f' % (if_flex - if_ext))
  print('Thumb min, max = %0.2f, %0.2f' % (thumb_ext, thumb_flex))
  print('Diff = %0.2f' % (thumb_flex - thumb_ext))
  #print('Index Finger Flexion')
  #print('min = %0.2f' % cd.get_min())
  #print('max = %0.2f' % cd.get_max())
  #print('avg = %0.2f' % cd.get_average())

  vd = []
  
  print(fd.get_data(0))
  print(fd.get_data(1))
  
  if fd.get_data(0).get_value(opt.column_name) <= fd.get_data(1).get_value(opt.column_name):
    direction = 1
  else:
    direction = -1
    
  # print('initial direction = %d' % (direction))

  vd.append(VelocityData(fd.get_data(0).get_value('timestamp'), fd.get_data(0).get_value(opt.column_name), direction))
  
  for i in range(1, fd.get_data_count() - 1):
    if not vd[-1].check_data(fd.get_data(i).get_value('timestamp'), fd.get_data(i).get_value(opt.column_name)):
      if vd[-1].start_time - vd[-1].end_time != 0:
        direction = direction * -1
        vd.append(VelocityData(fd.get_data(i).get_value('timestamp'), fd.get_data(i).get_value(opt.column_name), direction))
  
  vd[-1].enter_final_value(fd.get_data(fd.get_data_count()-1).get_value('timestamp'), fd.get_data(fd.get_data_count()-1 ).get_value(opt.column_name))

  print('record count = %d' % (fd.get_data_count()))
  print('velocity changes = %d' % len(vd))
  max_velocity = None
  max_distance = None
  max_time = None
  for entry in vd:
    if max_time is None or max_time.get_time() < entry.get_time():
      max_time = entry
    if max_distance is None or abs(max_distance.get_distance()) < abs(entry.get_distance()):
      max_distance = entry
    if max_velocity is None or abs(max_velocity.get_velocity()) < abs(entry.get_velocity()):
      max_velocity = entry
  print('max velocity = %0.2f, start time = %d, end time = %d' % (abs(max_velocity.get_velocity()), max_velocity.start_time, max_velocity.end_time))
  print('max time = %0.2f, start_time = %d, end_time = %d' % (max_time.get_time(), max_time.start_time, max_time.end_time))
  print('max distance = %0.2f' % (abs(max_distance.get_distance())))
  # sys.exit()
  
  print('record count = %d' % fd.get_data_count())
  print('max = %0.2f' % fd.get_max(opt.column_name))
  print('min = %0.2f' % fd.get_min(opt.column_name))
  print('avg = %0.2f' % fd.get_average(opt.column_name))
  print('std = %0.2f' % fd.get_std_dev(opt.column_name))
  print('max_step = %0.2f' % fd.get_max_step(opt.column_name))
  print('max_step_percent = %0.2f%%' % fd.get_max_step_percent(opt.column_name))
  histogram = fd.construct_histogram_data(opt.column_name, bucket_count=20)
  for h in histogram:
    print(h)
  average_set = fd.create_average_set(opt.column_name, 50)
  max_val = None
  min_val = None
  for v in average_set:
    if max_val is None or max_val < v:
      max_val = v
    if min_val is None or min_val > v:
      min_val = v
  print('max from average set = %0.2f' % (max_val))
  print('min from avreage set = %0.2f' % (min_val))
  
  sensor1_set = {}
  sensor2_set = {}
  
  
  
  #plt.plot(average_set)
  #plt.show()
  
  
if __name__ == '__main__':
  main()