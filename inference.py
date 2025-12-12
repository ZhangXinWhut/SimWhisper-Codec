import os
import argparse
import logging
import torch

from utils.helpers import set_logging, load_audio, save_audio, find_audio_files
from audiocodec.model import AudioCodec

if __name__ == "__main__":
    set_logging()
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="./config/SimWhisperCodec.yaml")
    parser.add_argument("--checkpoint_path", type=str, default="./weights/SimWhisperCodec.pt")
    parser.add_argument("--device", type=str, default="cuda")
    
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--input_dir", type=str, default="input_wavs")
    parser.add_argument("--output_dir", type=str, default="output_wavs")
    
    args = parser.parse_args()

    device = torch.device(args.device)

    ## Load codec model
    generator = AudioCodec.load_from_checkpoint(config_path=args.config_path, ckpt_path=args.checkpoint_path).to(device).eval()
    
    ## Find audios
    audio_paths = find_audio_files(input_dir=args.input_dir)
    
    ## Create output directory if not exists
    os.makedirs(args.output_dir, exist_ok=True)
    logging.info(f"Processing {len(audio_paths)} audio files, output will be saved to {args.output_dir}")

    with torch.no_grad():
        ## Process audios in batches
        batch_size = args.batch_size
        for i in range(0, len(audio_paths), batch_size):
            batch_paths = audio_paths[i:i + batch_size]
            logging.info(f"Processing batch {i // batch_size + 1}/{len(audio_paths) // batch_size + 1}, files: {batch_paths}")

            # Load audio files
            wav_list = [load_audio(path, target_sample_rate=generator.input_sample_rate).squeeze().to(device) for path in batch_paths]
            logging.info(f"Successfully loaded {len(wav_list)} audio files with lengths {[len(wav) for wav in wav_list]} samples")

            # Encode
            encode_result = generator.encode(wav_list, overlap_seconds=10)
            codes_list = encode_result["codes_list"]  # B * (nq, T)
            logging.info(f"Encoding completed, code lengths: {[codes.shape[-1] for codes in codes_list]}")
            # logging.info(f"{codes_list = }")

            # Decode
            decode_result = generator.decode(codes_list, overlap_seconds=10)
            syn_wav_list = decode_result["syn_wav_list"]  # B * (T,)
            logging.info(f"Decoding completed, generated waveform lengths: {[len(wav) for wav in syn_wav_list]} samples")

            # Save generated audios
            for path, syn_wav in zip(batch_paths, syn_wav_list):
                # 强制输出为 wav 格式，无论输入是什么格式
                input_filename = os.path.basename(path)
                output_filename = os.path.splitext(input_filename)[0] + '.wav'
                output_path = os.path.join(args.output_dir, output_filename)
                save_audio(output_path, syn_wav.cpu().reshape(1, -1), sample_rate=generator.output_sample_rate)
                logging.info(f"Saved generated audio to {output_path}")

        
    logging.info("All audio processing completed")