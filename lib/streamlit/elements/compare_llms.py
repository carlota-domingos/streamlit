# Copyright (c) Streamlit Inc. (2018-2022) Snowflake Inc. (2022-2025)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from streamlit.delta_generator import DeltaGenerator


class CompareLLMsMixin:
    def compare_llms(
        self: DeltaGenerator,
        *,
        model_a: str,
        model_b: str,
        get_response_a=None,
        get_response_b=None,
        history_key: str = "compare_llms_history",
        chatbox_placeholder: str = "Insert your question...",
    ) -> None:
        """Display a chatbox and, for each prompt, show LLM responses side by side with ratings.

        Parameters
        ----------
        model_a : str
            The name of the first model.
        model_b : str
            The name of the second model.
        get_response_a : callable or None
            Function that takes a prompt and returns the response from model_a.
        get_response_b : callable or None
            Function that takes a prompt and returns the response from model_b.
        history_key : str
            Session state key for chat history.
        chatbox_placeholder : str
            Placeholder text for the chat input.

        Example
        -------
        >>> def get_a(prompt):
        ...     return "A: " + prompt
        >>> def get_b(prompt):
        ...     return "B: " + prompt
        >>> st.compare_llms(
        ...     model_a="llama-3",
        ...     model_b="gemma-2",
        ...     get_response_a=get_a,
        ...     get_response_b=get_b,
        ... )
        """
        import streamlit as st

        history_state_key = f"{history_key}_history"
        chat_input_key = f"{history_key}_input"

        if history_state_key not in st.session_state:
            st.session_state[history_state_key] = []

        prompt = st.chat_input(chatbox_placeholder, key=chat_input_key)
        if prompt and get_response_a and get_response_b:
            response_a = get_response_a(prompt)
            response_b = get_response_b(prompt)
            st.session_state[history_state_key].append(
                {
                    "prompt": prompt,
                    "response_a": response_a,
                    "response_b": response_b,
                }
            )

        for entry in st.session_state[history_state_key]:
            with st.chat_message("user"):
                st.markdown(f"**Você:** {entry['prompt']}")
            with st.chat_message("assistant"):
                prompt_hash = hashlib.sha256(
                    f"{model_a}|{model_b}|{entry['prompt']}".encode()
                ).hexdigest()[:8]
                if "ratings" not in st.session_state:
                    st.session_state["ratings"] = {}

                col1, col2 = self.columns(2)
                with col1:
                    col1.subheader(model_a)
                    col1.markdown(entry["response_a"])
                    rating_a = col1.slider(
                        f"Evaluate the answer from {model_a}:",
                        min_value=1,
                        max_value=5,
                        step=1,
                        format="%d.",
                        value=st.session_state["ratings"].get(
                            f"{model_a}_{prompt_hash}_a", 3
                        ),
                        key=f"rating_{model_a}_{prompt_hash}_a",
                    )
                    st.session_state["ratings"][f"{model_a}_{prompt_hash}_a"] = rating_a

                with col2:
                    col2.subheader(model_b)
                    col2.markdown(entry["response_b"])
                    rating_b = col2.slider(
                        f"Evaluate the answer from {model_b}:",
                        min_value=1,
                        max_value=5,
                        step=1,
                        format="%d.",
                        value=st.session_state["ratings"].get(
                            f"{model_b}_{prompt_hash}_b", 3
                        ),
                        key=f"rating_{model_b}_{prompt_hash}_b",
                    )
                    st.session_state["ratings"][f"{model_b}_{prompt_hash}_b"] = rating_b


def compare_llms(
    *,
    model_a: str,
    model_b: str,
    get_response_a=None,
    get_response_b=None,
    history_key: str = "compare_llms_history",
    chatbox_placeholder: str = "Insert your question...",
) -> None:
    """Public API for compare_llms chat+ratings widget.

    See CompareLLMsMixin.compare_llms for parameter documentation.
    """
    import streamlit as st

    return CompareLLMsMixin.compare_llms(
        st,
        model_a=model_a,
        model_b=model_b,
        get_response_a=get_response_a,
        get_response_b=get_response_b,
        history_key=history_key,
        chatbox_placeholder=chatbox_placeholder,
    )
