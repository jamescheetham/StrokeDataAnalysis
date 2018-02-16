import Analyser, csv

def main():
  sample_info = { x : { y : [] for y in ['min', 'max'] } for x in ['index', 'thumb']}
  sample_info['index']['min'] = [6, 2, 40, 4]
  sample_info['index']['max'] = [16, 2, 40, 4]
  sample_info['thumb']['min'] = [6, 2, 40, 4]
  sample_info['thumb']['max'] = [26, 2, 40, 4]
  pilot_data = Analyser.Subject('pilot/170808', { 'digits' : ['thumb', 'index']})
  for f in ['calibration1', 'calibration2']:
    pilot_data.read_sample_file(f, 4, 5, 10)
    pilot_data.data_set.data[f].print_velocity_data('pilot/%s Velocity Data.csv' % f)
    for d in ['index', 'thumb']:
      for m in ['min', 'max']:
        pilot_data.generate_calibration_data(f, d, m, 10, sample_info[d][m][0], sample_info[d][m][1], sample_info[d][m][2], sample_info[d][m][3])
  pilot_data.generate_velocity_change_plots()
  pilot_data.generate_calibration_plot()
  pilot_data.calc_calibration_values()


  threshold = 1.0

  times = { 'calibration1' : [[33, 45], [73.2, 84], [113, 124]], 'calibration2': [[32,44], [72.1, 84], [112, 117]] }

  #values = { 'calibration1' : { 'index' : [], 'thumb' : [] }, 'calibration2': { 'index' : [], 'thumb' : [] } }
  values = { x : { y : { z : [] for z in ['Slow', 'Medium', 'Fast'] } for y in ['index', 'thumb'] } for x in ['calibration1', 'calibration2'] }

  f = open('pilot/170808/calibration_output.csv', 'w')
  csv_writer = csv.writer(f, delimiter=',', quotechar='"')
  csv_writer.writerow(['Calibration File', 'Digit', 'Typing Speed', 'Movements', 'Movements/Sec', 'Average Velocity (%ROM/s)', 'Average Magnitude (%ROM)'])

  for k, v in times.items():
    for d in ['index', 'thumb']:
      for i in range(len(v)):
        time_points = v[i]
        if i == 0:
          speed = 'Slow'
        elif i == 1:
          speed = 'Medium'
        elif i == 2:
          speed = 'Fast'
        print('%s - %s' % (k, d))
        print('Typing Speed between %0.2f and %0.2f' % (time_points[0], time_points[1]))
        movements = pilot_data.data_set.data[k].get_movement_count(d, time_points[0], time_points[1], threshold)
        values[k][d][speed].append(movements)
        values[k][d][speed].append(movements/(time_points[1] - time_points[0]))
        values[k][d][speed].append(pilot_data.data_set.data[k].get_movement_average_velocity(d, time_points[0], time_points[1], threshold)/pilot_data.ROM[d])
        values[k][d][speed].append(pilot_data.data_set.data[k].get_movement_average_magnitude(d, time_points[0], time_points[1], threshold)[2]/pilot_data.ROM[d])
        print(' - Movements = %d' % movements)
        print(' - per second = %0.2f' % values[k][d][speed][1])
        print(' - avg velocity = %0.2f' % values[k][d][speed][2])
        print(' - avg magnitude = %0.2f\n' % values[k][d][speed][3])
        csv_writer.writerow([k, d, speed] + values[k][d][speed])
  f.close()




if __name__ == '__main__':
  main()