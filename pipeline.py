# %%
from kfp import dsl, compiler
from kfp.dsl import Input, Output, Artifact


@dsl.component(packages_to_install=["pytube==12.1.2"])
def download_videos(
    videos_ids: Input[Artifact],
    audio_files: Output[Artifact],
    video_titles: Output[Artifact],
):
    from shutil import make_archive
    import tempfile
    from pytube import YouTube
    from pytube.exceptions import RegexMatchError
    import json

    with open(videos_ids.path, "r") as f:
        videos_ids = f.read().splitlines()

    tmpdirname = tempfile.TemporaryDirectory()
    video_titles_dict = {}
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
        video_titles_dict[video_id] = yt.title
    with open(video_titles.path, "w") as f:
        f.write(json.dumps(video_titles_dict))
    make_archive(audio_files.path, "gztar", tmpdirname.name)
    # audio_files.uri = audio_files.path + ".tar.gz"
    audio_files.metadata["name"] = "audio_files.tar.gz"
    audio_files.uri = audio_files.uri + ".tar.gz"


@dsl.component(
    base_image="davidnet/small-ffmpeg:v2",
    packages_to_install=["openai-whisper==20230124"],
)
def transcribe_audios(
    audio_files: Input[Artifact],
    video_titles: Input[Artifact],
    whisper_model: str,
    transcriptions: Output[Artifact],
):
    import shutil
    from pathlib import Path
    import whisper
    import torch
    import json

    print("Starting")
    with open(video_titles.path, "r") as f:
        video_titles_dict = json.load(f)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device}")
    # model = whisper.load_model("medium.en").to(device)
    model = whisper.load_model(whisper_model).to(device)
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
                "title": video_titles_dict[id[:-4]],
                "url": f"https://youtu.be/{id}",
                **{
                    "id": f"{id}-t{segments[j]['start']}",
                    "text": segment["text"].strip(),
                    "start": segment["start"],
                    "end": segment["end"],
                },
            }
            data.append(meta)
    with open(transcriptions.path, "w", encoding="utf-8") as fp:
        for line in data:
            json.dump(line, fp)
            fp.write("\n")

    # rename the file
    shutil.move(transcriptions.path, transcriptions.path + ".jsonl")
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


@dsl.component(
    base_image="pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime",
    packages_to_install=[
        "sentence-transformers==2.2.2",
        "pinecone-client==2.1.0",
        "tqdm==4.64.1",
    ],
)
def create_and_push_embeddings(
    transcriptions: Input[Artifact],
    index_id: str,
    window: int,
    stride: int,
    batch_size: int,
):
    import json
    from pathlib import Path
    from tqdm import tqdm
    from sentence_transformers import SentenceTransformer
    import pinecone

    print(transcriptions.path)
    # Read a .jsonl file
    data = []
    with open(transcriptions.path, "r", encoding="utf-8") as fp:
        for line in fp:
            data.append(json.loads(line))
    new_data = []

    # window = 6  # number of sentences to combine
    # stride = 3  # number of sentences to 'stride' over, used to create overlap

    for i in tqdm(range(0, len(data), stride)):
        i_end = min(len(data) - 1, i + window)
        if data[i]["title"] != data[i_end]["title"]:
            # in this case we skip this entry as we have start/end of two videos
            continue
        text = " ".join(
            [small_transcript["text"] for small_transcript in data[i:i_end]]
        )
        new_data.append(
            {
                "start": data[i]["start"],
                "end": data[i_end]["end"],
                "title": data[i]["title"],
                "text": text,
                "id": data[i]["id"],
                "url": data[i]["url"],
            }
        )
    model_id = "multi-qa-mpnet-base-dot-v1"
    model = SentenceTransformer(model_id, device=None)
    print(model)
    dim = model.get_sentence_embedding_dimension()
    pinecone.init(
        api_key="NO-IDEA-HOW-TO-SAFEGUARD",  # app.pinecone.io
        environment="us-east1-gcp",  # find next to API key
    )

    if index_id not in pinecone.list_indexes():
        pinecone.create_index(index_id, dim, metric="dotproduct")

    index = pinecone.Index(index_id)

    # loop through in batches of 64
    for i in tqdm(range(0, len(new_data), batch_size)):
        # find end position of batch (for when we hit end of data)
        i_end = min(len(new_data) - 1, i + batch_size)
        # extract the metadata like text, start/end positions, etc
        batch_meta = [
            {
                "text": new_data[x]["text"],
                "start": new_data[x]["start"],
                "end": new_data[x]["end"],
                "url": new_data[x]["url"],
                "title": new_data[x]["title"],
            }
            for x in range(i, i_end)
        ]
        # extract only text to be encoded by embedding model
        batch_text = [row["text"] for row in new_data[i:i_end]]
        # create the embedding vectors
        batch_embeds = model.encode(batch_text).tolist()
        # extract IDs to be attached to each embedding and metadata
        batch_ids = [row["id"] for row in new_data[i:i_end]]
        # 'upsert' (insert) IDs, embeddings, and metadata to index
        to_upsert = list(zip(batch_ids, batch_embeds, batch_meta))
        index.upsert(to_upsert)

    print(index.describe_index_stats())


# %%
@dsl.pipeline()
def pipeline(
    video_location: str = "https://raw.githubusercontent.com/charlesa101/youtube-whisper-sbert/main/urls-from-Community-notes",
    whisper_model: str = "medium.en",
    index_id: str = "kubeflow-search",
    window: int = 6,
    stride: int = 3,
    batch_size: int = 64,
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
        audio_files=download_task.outputs["audio_files"],
        whisper_model=whisper_model,
        video_titles=download_task.outputs["video_titles"],
    )
    transcribe_task.set_cpu_limit("2").set_memory_limit(
        "8G"
    ).add_node_selector_constraint("NVIDIA_TESLA_T4").set_gpu_limit("1")
    embedding_task = create_and_push_embeddings(
        transcriptions=transcribe_task.output,
        index_id=index_id,
        window=window,
        stride=stride,
        batch_size=batch_size,
    )
    embedding_task.set_cpu_limit("2").set_memory_limit(
        "8G"
    ).add_node_selector_constraint("NVIDIA_TESLA_T4").set_gpu_limit("1")


if __name__ == "__main__":
    compiler.Compiler().compile(pipeline, "pipeline.yaml")
