import ffmpeg

input_pattern = 'DAVIS/2017/trainval/JPEGImages/480p/dogs-jump/%05d.jpg'
output_file = 'output.mp4'
frame_rate = 30

ffmpeg.input(input_pattern, framerate=frame_rate).output(output_file).run()