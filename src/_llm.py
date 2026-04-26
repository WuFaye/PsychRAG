import os
import json
import asyncio
import requests
from openai import AsyncOpenAI, OpenAI
from base import BaseKVStorage
from _utils import compute_args_hash
from typing import List, Dict, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline


def _get_required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise EnvironmentError(f"Required environment variable '{name}' is not set.")
    return value


async def siliconflow_response_if_catch(
    user_prompt,
    system_prompt: Optional[str] = None,
    history_messages: Optional[List[Dict]] = None,
    model_name: str = "Qwen/Qwen2-7B-Instruct",
    **kwargs
) -> str:
    api_key = _get_required_env("SILICONFLOW_API_KEY")
    api_url = "https://api.siliconflow.cn/v1/chat/completions"
    model = model_name

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    history_messages = history_messages or []
    hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)

    messages.extend(history_messages)
    messages.append({"role": "user", "content": user_prompt})

    if hashing_kv is not None:
        args_hash = compute_args_hash(model, messages)
        cached = await hashing_kv.get_by_id(args_hash)
        if cached is not None:
            return cached["return"]

    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "max_tokens": 4096,
        "temperature": 0.7,
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    response = requests.post(api_url, json=payload, headers=headers).json()

    while "choices" not in response:
        print("\nChoices missing from response. Retrying...\n")
        response = requests.post(api_url, json=payload, headers=headers).json()

    result = response["choices"][0]["message"]["content"]

    if hashing_kv is not None:
        await hashing_kv.upsert({args_hash: {"return": result, "model": model}})

    print(response)
    return result


async def deepseek_response_if_catch(
    user_prompt,
    system_prompt: Optional[str] = None,
    history_messages: Optional[List[Dict]] = None,
    model_name: str = "deepseek-reasoner",
    **kwargs
) -> str:
    api_key = _get_required_env("DEEPSEEK_API_KEY")
    model = model_name

    client = AsyncOpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com",
        timeout=360,
    )

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    history_messages = history_messages or []
    hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)

    messages.extend(history_messages)
    messages.append({"role": "user", "content": user_prompt})

    if hashing_kv is not None:
        args_hash = compute_args_hash(model, messages)
        cached = await hashing_kv.get_by_id(args_hash)
        if cached is not None:
            return cached["return"]

    while True:
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                **kwargs,
            )
            break
        except json.JSONDecodeError:
            print("JSONDecodeError encountered. Retrying...")
            await asyncio.sleep(1)

    result = response.choices[0].message.content

    if hashing_kv is not None:
        await hashing_kv.upsert({args_hash: {"return": result, "model": model}})

    return result


def doubao_response(
    user_prompt: str,
    system_prompt: Optional[str] = None,
    history_messages: Optional[List[Dict]] = None,
) -> str:
    api_key = _get_required_env("DOUBAO_API_KEY")

    client = OpenAI(
        base_url="https://ark.cn-beijing.volces.com/api/v3/",
        api_key=api_key,
        timeout=180,
    )

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    history_messages = history_messages or []
    messages.extend(history_messages)
    messages.append({"role": "user", "content": user_prompt})

    completion = client.chat.completions.create(
        model="doubao-1-5-pro-256k-250115",
        messages=messages,
        stream=False,
    )

    return completion.choices[0].message.content


def deepseek_response(
    user_prompt: str,
    system_prompt: Optional[str] = None,
    history_messages: Optional[List[Dict]] = None,
    model_name: str = "deepseek-reasoner",
) -> Optional[str]:
    api_key = _get_required_env("DEEPSEEK_API_KEY")
    model = model_name

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com",
    )

    messages = []
    if system_prompt is not None:
        messages.append({"role": "system", "content": system_prompt})

    history_messages = history_messages or []
    messages.extend(history_messages)
    messages.append({"role": "user", "content": user_prompt})

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            stream=False,
            temperature=1.0,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error occurred: {e}")
        return None


def qwen_response(
    user_prompt: str,
    system_prompt: Optional[str] = None,
    history_messages: Optional[List[Dict]] = None,
    model_name: str = "Qwen/Qwen2-7B-Instruct",
) -> str:
    api_key = _get_required_env("SILICONFLOW_API_KEY")
    api_url = "https://api.siliconflow.cn/v1/chat/completions"
    model = model_name

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    history_messages = history_messages or []
    messages.extend(history_messages)
    messages.append({"role": "user", "content": user_prompt})

    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "max_tokens": 4096,
        "temperature": 0.7,
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    response = requests.post(api_url, json=payload, headers=headers).json()
    return response["choices"][0]["message"]["content"]


def siliconflow_response(
    model_name: str,
    user_prompt: str,
    system_prompt: Optional[str] = None,
    history_messages: Optional[List[Dict]] = None,
) -> str:
    api_key = _get_required_env("SILICONFLOW_API_KEY")
    api_url = "https://api.siliconflow.cn/v1/chat/completions"
    model = model_name

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    history_messages = history_messages or []
    messages.extend(history_messages)
    messages.append({"role": "user", "content": user_prompt})

    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "max_tokens": 4096,
        "temperature": 1.0,
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    response = requests.post(api_url, json=payload, headers=headers).json()
    print(response)

    return response["choices"][0]["message"]["content"]


# ---------------- Local Qwen2-7B inference ----------------

LOCAL_PIPELINE = None
LOCAL_TOKENIZER = None
LOCAL_MODEL = None


def _build_local_pipeline(model_name: str = "Qwen/Qwen2-7B"):
    global LOCAL_PIPELINE, LOCAL_TOKENIZER, LOCAL_MODEL

    if LOCAL_PIPELINE is not None:
        return LOCAL_PIPELINE

    LOCAL_TOKENIZER = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        local_files_only=True,
    )

    LOCAL_MODEL = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map={"": "cuda:5"},
        local_files_only=True,
    )

    LOCAL_PIPELINE = pipeline(
        "text-generation",
        model=LOCAL_MODEL,
        tokenizer=LOCAL_TOKENIZER,
        trust_remote_code=True,
    )

    return LOCAL_PIPELINE


def local_response(
    user_prompt: str,
    system_prompt: Optional[str] = None,
    history_messages: Optional[List[Dict]] = None,
    model_name: str = "Qwen/Qwen2-7B",
) -> str:
    history_messages = history_messages or []

    if system_prompt:
        prompt_text = f"System:\n{system_prompt}\n\nUser:\n{user_prompt}"
    else:
        prompt_text = user_prompt

    pipe = _build_local_pipeline(model_name)

    output = pipe(
        prompt_text,
        max_new_tokens=1024,
        do_sample=False,
        temperature=0.0,
    )

    return output[0]["generated_text"]


async def local_response_if_catch(
    user_prompt,
    system_prompt: Optional[str] = None,
    history_messages: Optional[List[Dict]] = None,
    model_name: str = "Qwen/Qwen2-7B",
    **kwargs
) -> Optional[str]:
    hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
    history_messages = history_messages or []

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    messages.extend(history_messages)
    messages.append({"role": "user", "content": user_prompt})

    model = model_name

    if hashing_kv is not None:
        args_hash = compute_args_hash(model, messages)
        cached = await hashing_kv.get_by_id(args_hash)
        if cached is not None:
            return cached["return"]

    try:
        result = await asyncio.to_thread(
            local_response,
            user_prompt,
            system_prompt,
            history_messages,
            model_name,
        )
    except Exception as e:
        print(f"local_response failed: {e}")
        return None

    if hashing_kv is not None:
        await hashing_kv.upsert({args_hash: {"return": result, "model": model}})

    return result