# node-extractor

## Explanation
In this project, a text-to-sequence-of-nodes application was developed for the Icari task. The project utilizes the Llama-7b-chat-hf language model, which has been effectively guided for the task through 'prompt' engineering. A simple Swagger FastAPI has been set up to make the model accessible for user testing. You can follow the setup and testing instructions below to try it out.

The core of this project was optimizing the well-known 'llama' language model through well-designed prompting. Exact instructions for the model were provided along with generic examples. This approach yielded the highest accuracy.

Note: As illustrated in 'data/data.json', more than 150 example prompts and their corresponding ground truth sequences of nodes are manually created to train(fine tune) a language model for this task. Although I chose the DistilBert model (also gave a quick try to T5, GPT-2 language models but could not try some of them and others because of not having enough vram on my device)for this task, processing these data and fine-tuning the model did not achieve satisfactory results. The primary issue was the difficulty in augmenting or expanding the dataset, as the data must be genuine and thoroughly validated. While the trained model was successful in extracting nodes with high accuracy, it struggled with ordering these nodes correctly. Due to the lack of sufficient ground truth data and time, I shifted focus to using a well-trained language model and refined it with effective prompting. However, if you wish to review the data and code, please refer to 'data/data.json' and 'distilbert_trainer.py'.

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


## Unit&Integration Test
You can run unit and integration tests by calling the following commands.
```bash
pytest test_action_extractor.py
pytest test_integration.py
```