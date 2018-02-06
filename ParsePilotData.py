import Analyser

def main():
  pilot_data = Analyser.Subject('pilot/170808', { 'digits' : ['thumb', 'index']})
  for f in ['calibration1', 'calibration2']:
    pilot_data.read_sample_file(f, 4, 5, 10)
    pilot_data.data_set.data[f].print_velocity_data('pilot/%s Velocity Data.csv' % f)
  pilot_data.generate_velocity_change_plots()


  threshold = 1.0

  times = { 'calibration1' : [[33, 45], [73.2, 84], [113, 124]], 'calibration2': [[32,44], [72.1, 84], [112, 117]] }

  for k, v in times.items():
    for d in ['index', 'thumb']:
      for time_points in v:
        print('%s - %s' % (k, d))
        print('Typing Speed between %0.2f and %0.2f' % (time_points[0], time_points[1]))
        movements = pilot_data.data_set.data[k].get_movement_count(d, time_points[0], time_points[1], threshold)
        print(' - Movements = %d' % movements)
        print(' - per second = %0.2f' % (movements/(time_points[1] - time_points[0])))
        print(' - avg velocity = %0.2f' % (pilot_data.data_set.data[k].get_movement_average_velocity(d, time_points[0], time_points[1], threshold)))
        print(' - avg magnitude = %0.2f\n' % (pilot_data.data_set.data[k].get_movement_average_magnitude(d, time_points[0], time_points[1], threshold)))

if __name__ == '__main__':
  main()