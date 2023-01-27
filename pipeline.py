# %%
from kfp import dsl, compiler
from kfp.dsl import Input, Output, Artifact


@dsl.component(packages_to_install=["pytube==12.1.2"])
def download_videos(videos_ids: Input[Artifact], audio_files: Output[Artifact]):
    from shutil import make_archive
    import tempfile
    from pytube import YouTube
    from pytube.exceptions import RegexMatchError

    with open(videos_ids.path, "r") as f:
        videos_ids = f.read().splitlines()

    tmpdirname = tempfile.TemporaryDirectory()
    for video_id in videos_ids:
        url = f"https://youtu.be/{video_id}"
        try:
            yt = YouTube(url)
        except RegexMatchError:
            print(f"RegexMatchError for '{url}'")
            continue
        itag = None
        # we only want audio files
        files = yt.streams.filter(only_audio=True)
        for file in files:
            # from audio files we grab the first audio for mp4 (eg mp3)
            if file.mime_type == "audio/mp4":
                itag = file.itag
                break
            if itag is None:
                # just incase no MP3 audio is found (shouldn't happen)
                print("NO MP3 AUDIO FOUND")
                continue
        # get the correct mp3 'stream'
        stream = yt.streams.get_by_itag(itag)
        # downloading the audio
        stream.download(output_path=tmpdirname.name, filename=f"{video_id}.mp3")
    make_archive(audio_files.path, "gztar", tmpdirname.name)
    # audio_files.uri = audio_files.path + ".tar.gz"
    audio_files.metadata["name"] = "audio_files.tar.gz"
    audio_files.uri = audio_files.uri + ".tar.gz"


@dsl.component(
    base_image="davidnet/small-ffmpeg:v1",
    packages_to_install=["openai-whisper==20230124"],
)
def transcribe_audios(audio_files: Input[Artifact], transcriptions: Output[Artifact]):
    import shutil
    from pathlib import Path
    import whisper
    import torch
    import json

    print("Starting")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisper.load_model("base").to(device)
    print("model loaded")
    audio_extraction_dir = Path("audio_files")
    audio_extraction_dir.mkdir()
    shutil.unpack_archive(audio_files.path, extract_dir=audio_extraction_dir.absolute())
    # get a list of all mp3 files in the audio_extraction_dir directory
    audio_files_list = list(audio_extraction_dir.glob("*.mp3"))
    data = []
    for audio_file in audio_files_list:
        id = audio_file.name
        print("Transcribing ", id)
        result = model.transcribe(audio_file.absolute().as_posix())
        print("processed ", id)
        segments = result["segments"]
        for j, segment in enumerate(segments):
            # merge segments data and videos_meta data
            meta = {
                **{
                    "id": f"{id}-t{segments[j]['start']}",
                    "text": segment["text"].strip(),
                    "start": segment["start"],
                    "end": segment["end"],
                }
            }
            data.append(meta)
    with open(transcriptions.path, "w", encoding="utf-8") as fp:
        for line in data:
            json.dump(line, fp)
            fp.write("\n")

    transcriptions.uri = transcriptions.uri + ".jsonl"


@dsl.component(packages_to_install=["requests==2.28.2"])
def download_and_parse(urls: Input[Artifact], videos_ids: Output[Artifact]):
    import requests
    from urllib import parse

    print("Obtaining ", urls.uri)
    r = requests.get(urls.uri)
    raw_data = r.content.decode("utf-8").strip()
    # Extract v files https://stackoverflow.com/questions/5074803/retrieving-parameters-from-a-url
    yt_videos_id = [
        parse.parse_qs(parse.urlparse(elem).query)["v"][0] for elem in raw_data.split()
    ]
    with open(videos_ids.path, "w") as f:
        f.write("\n".join(yt_videos_id))


@dsl.pipeline()
def pipeline(
    video_location: str = "https://raw.githubusercontent.com/charlesa101/youtube-whisper-sbert/main/urls-from-Community-notes",
):
    importer_task = dsl.importer(
        artifact_uri=video_location,
        artifact_class=dsl.Artifact,
        reimport=False,
    )
    reader_task = download_and_parse(urls=importer_task.output)
    download_task = download_videos(videos_ids=reader_task.output)
    # TODO (davidnet): How to ask for an accelerator?
    transcribe_task = transcribe_audios(
        audio_files=download_task.output
    ).set_memory_limit("32G")


if __name__ == "__main__":
    compiler.Compiler().compile(pipeline, "pipeline.yaml")

# %%
