function frames = get_movie_frames(movie_pointer,wf)
%Import relevant movie frames
fprintf('Loading movie\r')
mov = VideoReader(movie_pointer);
frames = zeros(mov.Height,mov.Width,3,numel(wf),'uint8');
fwf = wf .* ceil(mov.FrameRate);
for idx = 1:numel(wf),
    fprintf('Reading frame %i\r',wf(idx));
    frames(:,:,:,idx) = read(mov, [fwf(idx),fwf(idx)]);
end