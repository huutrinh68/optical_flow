ffmpeg -r 30 -i frame_num_%05d.png -vcodec libx264 -pix_fmt yuv420p -r 60 out.mp4
