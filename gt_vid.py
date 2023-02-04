from pathlib import Path
import imageio
import numpy as np
import json
import argparse
import ffmpeg

class VideoWriter:
    """
    fps - frames per second
    hdr_mode - encode HDR moview
    codec - 'vp9' (works with Chrome) or 'h265' (everything else) - currently used only in the HDR mode
    """
    def __init__(self, fname, fps=24, hdr_mode=False, codec='h265', verbose=False):
        self.fname = fname
        self.fps = fps
        self.verbose = verbose
        self.hdr_mode = hdr_mode
        self.process = None
        #self.bit_depth = 10
        self.codec = codec
    """
    Write a frame stored as numpy WxHxC matric to a video file. The frame must be in the right, display-encoded colour space:
    BT.709 + sRGB nonlinearity for SDR
    BT.2020 + PQ for HDR
    """
    def write_frame_rgb(self, rgb):
        H, W, C = rgb.shape        
        if C == 1:
            rgb = np.concatenate([rgb]*3, -1)
        if self.process is None:
            if self.hdr_mode:
                if self.codec == 'h265':
                    self.process = (ffmpeg
                            .input('pipe:', format='rawvideo', pix_fmt='rgb48le', s='{}x{}'.format(W, H), r=self.fps, colorspace="bt2020nc", color_primaries="bt2020", color_trc="smpte2084")
                            .output(self.fname, pix_fmt='yuv420p10le', crf=12, vcodec='libx265', **{'x265-params': 'hdr-opt=1:repeat-headers=1:colorprim=bt2020:transfer=smpte2084:colormatrix=bt2020nc:master-display=G(0,0)B(0,0)R(0,0)WP(0,0)L(0,0):max-cll=0,0', 'preset': 'fast'} )
                            .overwrite_output()
                            .global_args( '-hide_banner')
                            .global_args( '-preset', 'fast' )
                            .global_args( '-loglevel', 'info' if self.verbose else 'quiet')
                            .run_async(pipe_stdin=True)
                            )
                elif self.codec == 'vp9':
                    self.process = (ffmpeg
                            .input('pipe:', format='rawvideo', pix_fmt='rgb48le', s='{}x{}'.format(W, H), r=self.fps, colorspace="bt2020nc", color_primaries="bt2020", color_trc="smpte2084")
                            .output(self.fname, pix_fmt='yuv420p10le', crf=10, vcodec='libvpx-vp9', **{'color_primaries': '9', 'color_trc': '16', 'colorspace': '9', 'color_range': '1', 'profile:v': '2' , 'preset': 'fast', 'b:v': '0' } )
                            .overwrite_output()
                            .global_args( '-hide_banner')
                            .global_args( '-preset', 'fast' )
                            .global_args( '-loglevel', 'info' if self.verbose else 'quiet')
                            .run_async(pipe_stdin=True)
                            )
                else:
                    raise RuntimeError( 'Unknown codec' )
            else:
                self.process = (ffmpeg
                        .input('pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format(W, H), r=self.fps)
                        .output(self.fname, pix_fmt='yuv444p', crf=10 )
                        .overwrite_output()
                        .global_args( '-hide_banner')
                        .global_args( '-loglevel', 'info' if self.verbose else 'quiet')
                        .run_async(pipe_stdin=True)
                )
        
        #ffmpeg.compile(self.process)
        if self.hdr_mode:
            self.process.stdin.write(
                (rgb * (2**16-1)).astype(np.uint16).tobytes()
                )            
        else:
            if rgb.dtype == np.uint8:
                self.process.stdin.write( rgb.tobytes() )
            else:
                self.process.stdin.write(
                    (rgb * 255.0)
                    .astype(np.uint8)
                    .tobytes()
                )
    # Delete or close if program was interrupted
    def __del__(self):
        self.close()        
    def close(self):
        if not self.process is None:
            self.process.stdin.close()
            self.process.wait()        
            self.process = None
    def __enter__(self):
        return self
    def __exit__(self, type, value, tb):
        self.close()

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str)

args = parser.parse_args()


scene_path = Path(args.data_path)

with (scene_path / 'split.json').open('r') as f:
    split_json = json.load(f)
    i_imgs = np.arange(split_json['n_imgs'])
    train_mask = np.array(split_json['id_is_train'])
    i_train = i_imgs[train_mask]
    i_test = i_imgs[~train_mask]

img_dir = scene_path / 'images'
test_gts = []

vw = VideoWriter(str(scene_path / 'test_gt.mp4'))

for i, img_path in enumerate(sorted(list(img_dir.glob('*')))):
    if not train_mask[i]:
        # test_gts.append(imageio.imread(str(img_path)))
        vw.write_frame_rgb(imageio.imread(str(img_path)))

# imageio.mimwrite(str(scene_path / 'test_gt.mp4'), test_gts, fps=30, quality=8)
