from Analyser import *
import csv


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
    with open('%s/%s_statistical_analysis_summary.csv' % (p, p)) as f:
      csv_reader = csv.reader(f, quotechar='"', delimiter=',')
      next(csv_reader) #ignore the header row
      line = next(csv_reader) #get the digits row
      digits = line[1].split(',')
      for i in range(3):
        next(csv_reader)
      for d in digits:
        line = next(csv_reader)
        max = float(line[1])
        line = next(csv_reader)
        min = float(line[1])
        s.max_flexion[d] = max
        s.max_extension[d] = min
      next(csv_reader)
      for d in digits:
        line = next(csv_reader)
        s.average_movement[d] = float(line[1])
        for i in range(2):
          next(csv_reader)
        line = next(csv_reader)
        s.movement_groupings.update( { d : [int(x) for x in line] } )
        next(csv_reader)
      next(csv_reader)
      for d in digits:
        line = next(csv_reader)
        s.average_velocity[d] = float(line[1])
        for i in range(2):
          next(csv_reader)
        line = next(csv_reader)
        s.velocity_groupings.update( { d : [int(x) for x in line] } )
        next(csv_reader)
      next(csv_reader)
      for d in digits:
        line = next(csv_reader)
        s.total_idle_time[d] = float(line[1][:-1])
        line = next(csv_reader)
        s.longest_idle_time[d] = float(line[1])
    subjects.append(s)



if __name__ == '__main__':
  main()