# Llama3 Model and Datasets for 4 Nordic Indigenous Languages

These Nordic indigenous (minority) languages carry rich cultural identity and heritage. Due to the scarcity of training data for these languages, we are releasing curated datasets and a lightweight Llama-based model. Our models are designed to run on low-cost consumer hardware, making them accessible even in environments with limited internet speed or GPU performance.

Slide
https://docs.google.com/presentation/d/1TGG8hRdhB4GPJd6iuzSZpArUbJ4FZoRMhFAJx_iaNcw/edit#slide=id.g32bc2cff546_0_0

Video
https://www.youtube.com/watch?v=R9-jJjCbEjg

## How to Run

Below are the instructions for running the models using `vllm`. You can run the merged model or any of the individual models by following these steps.

```bash
# Install vllm if you haven't already
pip install vllm

# Serve the merged model (Sami, Kven, Faroese, Kalaallisut)
vllm serve OpenGenerativeAI/Llama-3.2-3B-Sami_Kven_Faroese_Kalaallisut-Instruct-16bit

# Check the available models
curl http://localhost:8000/v1/models

# Send a completion request (example using the merged model)
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
          "model": "OpenGenerativeAI/Llama-3.2-3B-Sami_Kven_Faroese_Kalaallisut-Instruct-16bit",
          "prompt": "Davvi: Mii lea guovdilat davvirikkarášis vuosttaš eahket eará sániid numermalaš árvvu, jus galgá vuosttaš sáni.",
          "max_tokens": 700,
          "temperature": 0
        }'

```

To run any of the individual models, simply replace the model identifier in the command above:

-   **For Kven:**
    
    ```bash
    vllm serve OpenGenerativeAI/Llama-3.2-3B-Kven-Instruct-16bit
    
    ```
    
-   **For Kalaallisut:**
    
    ```bash
    vllm serve OpenGenerativeAI/Llama-3.2-3B-Kalaallisut-Instruct-16bit
    
    ```
    
-   **For Faroese:**
    
    ```bash
    vllm serve OpenGenerativeAI/Llama-3.2-3B-Faroese-Instruct-16bit
    
    ```
    
-   **For Sami:**
    
    ```bash
    vllm serve OpenGenerativeAI/Llama-3.2-3B-Sami-Instruct-16bit
    
    ```

## Languages Covered

### Sami Languages
- **Who Speaks Them:** The Sami people, who inhabit parts of northern Norway, Sweden, Finland, and Russia’s Kola Peninsula.
- **Varieties:** There isn’t a single “Sami language” but rather a group of closely related languages. The most widely spoken is Northern Sami, with other recognized varieties including Lule Sami, Southern Sami, Inari Sami, and Skolt Sami.
- **Status:** Official recognition and protection exist in several Nordic countries, with ongoing efforts to revitalize and promote these languages.

### Greenlandic (Kalaallisut)
- **Who Speaks It:** The Kalaallit (Greenlandic Inuit) of Greenland, an autonomous territory within the Kingdom of Denmark.
- **Language Family:** Eskimo-Aleut.
- **Status:** The official language of Greenland, playing a central role in the region’s cultural identity.

### Faroese
- **Who Speaks It:** The people of the Faroe Islands, another autonomous territory within the Kingdom of Denmark.
- **Language Family:** North Germanic, closely related to Icelandic and, to some extent, Norwegian.
- **Status:** The native language of the Faroese people and a significant aspect of their cultural heritage.

### Kven
- **Who Speaks It:** The Kven people in northern Norway, descendants of Finnish immigrants.
- **Language Family:** Closely related to Finnish.
- **Status:** Recognized as a minority language in Norway. While its classification as “indigenous” can vary in discussions, it is an important part of the Nordic linguistic landscape.

## Datasets

We have curated individual datasets for each language, as well as a merged dataset that includes all four languages. All datasets are hosted on [Hugging Face](https://huggingface.co/):

- **Merged Dataset (Sami, Kven, Faroese, Kalaallisut):**  
  [Llama3.3_Distil_Open_Sami_Kven_Faroese_Kalaallisut_Merged](https://huggingface.co/datasets/OpenGenerativeAI/Llama3.3_Distil_Open_Sami_Kven_Faroese_Kalaallisut_Merged)

- **Kalaallisut Dataset:**  
  [Llama3.3_Distil_Open_Kalaallisut](https://huggingface.co/datasets/OpenGenerativeAI/Llama3.3_Distil_Open_Kalaallisut)

- **Faroese Dataset:**  
  [Llama3.3_Distil_Open_Faroese](https://huggingface.co/datasets/OpenGenerativeAI/Llama3.3_Distil_Open_Faroese)

- **Kven Dataset:**  
  [Llama3.3_Distil_Open_Kven](https://huggingface.co/datasets/OpenGenerativeAI/Llama3.3_Distil_Open_Kven)

- **Sami Dataset:**  
  [Llama3.3_Distil_Open_Sami](https://huggingface.co/datasets/OpenGenerativeAI/Llama3.3_Distil_Open_Sami)

## Models

Our Llama-based models have been optimized for lower resource environments. They are available on Hugging Face:

- **Merged Model (Sami, Kven, Faroese, Kalaallisut):**  
  [Llama-3.2-3B-Sami_Kven_Faroese_Kalaallisut-Instruct-16bit](https://huggingface.co/OpenGenerativeAI/Llama-3.2-3B-Sami_Kven_Faroese_Kalaallisut-Instruct-16bit)

- **Kalaallisut Model:**  
  [Llama-3.2-3B-Kalaallisut-Instruct-16bit](https://huggingface.co/OpenGenerativeAI/Llama-3.2-3B-Kalaallisut-Instruct-16bit)

- **Faroese Model:**  
  [Llama-3.2-3B-Faroese-Instruct-16bit](https://huggingface.co/OpenGenerativeAI/Llama-3.2-3B-Faroese-Instruct-16bit)

- **Kven Model:**  
  [Llama-3.2-3B-Kven-Instruct-16bit](https://huggingface.co/OpenGenerativeAI/Llama-3.2-3B-Kven-Instruct-16bit)

- **Sami Model:**  
  [Llama-3.2-3B-Sami-Instruct-16bit](https://huggingface.co/OpenGenerativeAI/Llama-3.2-3B-Sami-Instruct-16bit)

