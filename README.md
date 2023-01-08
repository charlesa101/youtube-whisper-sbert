# Kubeflow Tribal Knowledge

Kubeflow Tribal Knowledge is an open source machine learning model that provides answers to questions about Kubeflow. The answers are provided from training a Natural Language Processing (NLP) model on the transcriptions of Kubeflow Community Meeting recordings. The goal is to supplement the Kubeflow documentation with an easy way to find current information on features that are in discussion, development or have limited documentation.

The model is based on the implementation defined in this post,https://www.pinecone.io/learn/openai-whisper/, with some modifications for Kubeflow.  The Kubeflow content is provided by transcribing recordings in the Kubeflow Community Youtube Channel.

The document proposes an implementation using these components:
1) Pytube to download MP3s from Youtube channel
2) Whisper for transcoding into 30 second segments
3) Sentence BERT (SBERT) for sentence transformer and transcript embeddings
4) Pinecone for storing 30 second segments with timestamps, and youtube channel info
5) Gradio for User Interface (question input, answer output)

And the workflow is shown in the diagram below:

<img width="895" alt="Screen Shot 2023-01-05 at 11 49 53 AM" src="https://user-images.githubusercontent.com/10553232/210847149-ad9d2172-bcb1-4d81-a5d6-8ce2814066e0.png">


The Kubeflow team is working to simplify this workflow, which includes integrating the dependencies so that they will easily run in Kubeflow.
