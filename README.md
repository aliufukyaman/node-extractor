# node-extractor

## Explanation
In this project, a text-to-sequence-of-nodes project was developed for Icari task. The project uses the Llama-7b-chat-hf language model. This model has been directed for the task as desired with the help of prompting. A simple Swagger FastAPI has been set up to make the model available to the user test. You can follow the setup and testing steps below to try it out.


## Building and Running:
In order to build the docker container, you can use following docker commands:

```bash
docker build -t node-extractor-image .
```

To run the model, use the following command:
```bash
docker run   --rm   --gpus all   --ipc=host   -p 8080:80   -v ~/.cache/huggingface/hub:/data   -e HF_API_TOKEN=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   ghcr.io/huggingface/text-generation-inference:0.9   --hostname 0.0.0.0   --model-id meta-llama/Llama-2-7b-chat-hf   --quantize bitsandbytes   --num-shard  1
```

To run the simple swagger UI of FastAPI, run the following command:
```bash
docker run -it --add-host=host.docker.internal:host-gateway --name node-extractor -p 8081:8081 node-extractor-image
```

## Usage of UI
After applying the previous building and running steps successfully, you will be able to access the UI by http://localhost:8081/

When you reached the link above, you will see a screen as seen below. Click the down arrow on right side of blue area of get method.
![Step1](assets/1.png)


Enter the input sentence to the 'sentence' textbox as seen below. Then clikck 'Execute' button to send the request.
![Step2](assets/2.png)


You will see the result of the request on 'Response body' part at bottom side of the window.
![Step3](assets/3.png)