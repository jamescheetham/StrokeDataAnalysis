import csv
from datetime import datetime
from optparse import OptionParser

class File_Data:
  def __init__(self, filename, data_col1 = 1, data_col2 = 2):
    self.filename = filename
    self.data = []
    self.read_data(data_col1, data_col2)
    
  def read_data(self, data_col1, data_col2):
    with open(self.filename, 'r') as f:
      trial_data = csv.reader(f, delimiter=',', quotechar='"')
      for line_data in trial_data:
        self.data.append(Data_Entry(line_data, data_col1, data_col2))
        #print(self.data[-1])
  
class Data_Entry:
  def __init__(self, line_data, data_col1, data_col2):
    self.timestamp = line_data[0]
    self.channel1 = line_data[data_col1]
    self.channel2 = line_data[data_col2]
    
  def __str__(self):
    return '%s,%s,%s' % (self.timestamp, self.channel1, self.channel2)

class Stroke_Detail:
  def __init__(self, stroke_date, stroke_type, affected_side, mas7_score, mas8_score):
    self.stroke_date = stroke_date
    self.stroke_type = stroke_type
    self.affected_side = affected_side
    self.mas7_score = mas7_score
    self.mas8_score = mas8_score
  
  def __str__(self):
    output_str = 'Stroke Date: %s\n' % (self.stroke_date.strftime('%d/%m/%Y'))
    output_str += 'Stroke Type: %s\n' % (self.stroke_type)
    output_str += 'Affected Side: %s\n' % (self.affected_side)
    output_str += 'MAS7 Score: %s\n' % (self.mas7_score)
    output_str += 'MAS8 Score: %s\n' % (self.mas8_score)
    return output_str

class Patient:
  def __init__(self, date, ident, age, gender, dominant_hand, tested_side, stroke_details=None):
    self.date = date
    self.patient_type = 'Stroke' if ident.startswith('STR') else 'Control'
    self.ident = ident
    self.age = age
    self.gender = gender
    self.dominant_hand = dominant_hand
    self.tested_side = tested_side
    self.stroke_details = stroke_details
    self.initial_calibration = None
    self.final_calibration = None
    self.trial_data = None
    
  def add_initial_calibration(self, calibration_file):
    self.initial_calibration = File_Data(calibration_file)
  
  def add_final_calibration(self, calibration_file):
    self.final_calibration = File_Data(calibration_file)
  
  def add_trial_data(self, calibration_file):
    self.trial_data = calibration_file
    
  def __str__(self):
    output_str = 'ID: %s\n' % (self.ident)
    output_str = 'Subject Type: %s\n' % (self.patient_type)
    if self.patient_type == 'Stroke':
      output_str += str(self.stroke_details)
    output_str += 'Age: %s\n' % (self.age)
    output_str += 'Gender: %s\n' % (self.gender)
    output_str += 'Dominant Hand: %s\n' % (self.dominant_hand)
    output_str += 'Tested Hand: %s\n' % (self.tested_side)
    return output_str
    
class Subject_Data:
  DATE_FORMAT = '%d/%m/%y'
  SUBJECT_DATA_PREFIX = 'Sensor Data '
  def __init__(self, folder_name):
    self.folder_name = folder_name
    self.log_file = folder_name + '/log.txt'
    self.process_log_file()
    self.read_calibration('initial', self.log_data['calibration1'])
    self.read_calibration('final', self.log_data['calibration2'])
    
  def process_log_file(self):
    self.log_data = {}
    with open(self.log_file, 'r') as f:
      for line_data in f:
        if line_data.strip() != '':
          data = [x.strip() for x in line_data.split(':')]
          self.log_data.update( { data[0] : data[1] } )
    if self.log_data['sub'].startswith('STR'):
      stroke_details = Stroke_Detail(datetime.strptime(self.log_data['stroke_date'], Subject_Data.DATE_FORMAT),
                                     self.log_data['stroke_type'],
                                     self.log_data['side_tested'],
                                     self.log_data['mas_score7'],
                                     self.log_data['mas_score8'])
    else:
      stroke_details = None
    self.patient = Patient(datetime.strptime(self.log_data['date'], Subject_Data.DATE_FORMAT),
                           self.log_data['sub'],
                           self.log_data['age'],
                           self.log_data['sex'],
                           self.log_data['hand_dom'],
                           self.log_data['side_tested'],
                           stroke_details)
        
  def read_calibration(self, calibration_type, file_suffix):
    if calibration_type == 'initial':
      self.patient.add_initial_calibration('%s/%s%s.csv' % (self.folder_name, Subject_Data.SUBJECT_DATA_PREFIX, file_suffix))
    elif calibration_type == 'final':
      self.patient.add_final_calibration('%s/%s%s.csv' % (self.folder_name, Subject_Data.SUBJECT_DATA_PREFIX, file_suffix))
      
  def __str__(self):
    output_str = ''
    for k, v in self.log_data.items():
      output_str += '%s: %s\n' % (k, v)
    output_str += '\nPatient Details\n%s' % (self.patient)
    return output_str
      
def main():
  parser = OptionParser()
  parser.add_option('-f', '--folder', type='string', action='store', dest='folder')
  
  (opt, arg) = parser.parse_args()
  
  if opt.folder is None:
    parser.error('Please enter a folder name')
    
  sd = Subject_Data(opt.folder)
  print(sd)
  
if __name__ == '__main__':
  main()