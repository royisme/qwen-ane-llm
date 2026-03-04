import ctypes
import json
import os
import platform
from typing import List, Dict, Any, Tuple, Optional

# Define C structures matching include/ane_lm/ane_lm_c.h
class ANEResponse(ctypes.Structure):
    _fields_ = [
        ("text", ctypes.c_char_p),
        ("token", ctypes.c_int),
        ("prompt_tokens", ctypes.c_int),
        ("prompt_tps", ctypes.c_float),
        ("generation_tokens", ctypes.c_int),
        ("generation_tps", ctypes.c_float),
    ]

# Callback type
# typedef void (*ane_callback_t)(ane_response_t* resp, void* user_data);
ANE_CALLBACK = ctypes.CFUNCTYPE(None, ctypes.POINTER(ANEResponse), ctypes.c_void_p)

class ANEBindingAdapter:
    def __init__(self, lib_path: str, model_dir: str):
        if not os.path.exists(lib_path):
            raise FileNotFoundError(f"Shared library not found at {lib_path}")
        
        self.lib = ctypes.CDLL(lib_path)
        
        # Define function signatures
        self.lib.ane_load_model.restype = ctypes.c_void_p
        self.lib.ane_load_model.argtypes = [ctypes.c_char_p, ctypes.c_bool]
        
        self.lib.ane_load_tokenizer.restype = ctypes.c_void_p
        self.lib.ane_load_tokenizer.argtypes = [ctypes.c_char_p]
        
        self.lib.ane_generate.restype = None
        self.lib.ane_generate.argtypes = [
            ctypes.c_void_p,      # model
            ctypes.c_void_p,      # tokenizer
            ctypes.c_char_p,      # json_messages
            ctypes.c_int,         # max_tokens
            ctypes.c_float,       # temperature
            ctypes.c_float,       # repetition_penalty
            ctypes.c_bool,        # enable_thinking
            ctypes.c_bool,        # reset_context
            ANE_CALLBACK,         # callback
            ctypes.c_void_p       # user_data
        ]
        
        self.lib.ane_free_model.argtypes = [ctypes.c_void_p]
        self.lib.ane_free_tokenizer.argtypes = [ctypes.c_void_p]

        # Load model and tokenizer once
        print(f"Loading model and tokenizer from {model_dir}...")
        model_dir_b = model_dir.encode('utf-8')
        self.model = self.lib.ane_load_model(model_dir_b, True)
        self.tokenizer = self.lib.ane_load_tokenizer(model_dir_b)
        
        if not self.model or not self.tokenizer:
            raise Exception("Failed to load model or tokenizer via shared library.")
        print("Model loaded successfully.")

    def __del__(self):
        if hasattr(self, 'model') and self.model:
            self.lib.ane_free_model(self.model)
        if hasattr(self, 'tokenizer') and self.tokenizer:
            self.lib.ane_free_tokenizer(self.tokenizer)

    def generate(self, messages: List[Dict[str, str]], temperature: float = 0.7, max_tokens: int = 512) -> Tuple[str, Dict[str, Any]]:
        full_text = []
        last_stats = {}
        
        def callback_func(resp_ptr, user_data):
            resp = resp_ptr.contents
            if resp.token != -1:
                if resp.text:
                    full_text.append(resp.text.decode('utf-8'))
            else:
                # Last packet with final stats
                last_stats['prompt_tokens'] = resp.prompt_tokens
                last_stats['generation_tokens'] = resp.generation_tokens
                last_stats['prompt_tps'] = resp.prompt_tps
                last_stats['generation_tps'] = resp.generation_tps

        c_callback = ANE_CALLBACK(callback_func)
        
        json_msg = json.dumps(messages).encode('utf-8')
        
        self.lib.ane_generate(
            self.model,
            self.tokenizer,
            json_msg,
            max_tokens,
            temperature,
            1.2,    # repetition_penalty
            False,  # enable_thinking
            False,  # reset_context
            c_callback,
            None
        )
        
        return "".join(full_text), last_stats
