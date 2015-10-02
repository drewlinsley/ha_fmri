frame_ind = 4215;
movie_path = '~/';%put the path on your computer to the movie here
mov = VideoReader([movie_path '/Home_Alone_2_PG.m4v']); %loads the movie information including frame rate and # of frames
frame = read(mov, [frame_ind frame_ind]);%you need to input a frame range (so in this case we repeat the frame twice)

figure; imagesc(frame)