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

"""compare_llms widget unit tests for chat-based API."""

import hashlib

import streamlit as st
from tests.delta_generator_test_case import DeltaGeneratorTestCase


class CompareLLMsTest(DeltaGeneratorTestCase):
    """Test the compare_llms widget (chat-based API)."""

    def setUp(self):
        super().setUp()
        st.session_state.clear()

    def simulate_chat(self, prompt, model_a="A", model_b="B", history_key=None):
        def get_a(p):
            return f"A: {p}"

        def get_b(p):
            return f"B: {p}"

        st.session_state["_streamlit_chat_input"] = prompt
        if history_key is None:
            history_key = f"compare_llms_history_{prompt.replace(' ', '_')}"
        st.compare_llms(
            model_a=model_a,
            model_b=model_b,
            get_response_a=get_a,
            get_response_b=get_b,
            history_key=history_key,
        )
        return history_key

    def test_compare_llms_renders(self):
        history_key = "test_renders"
        history_state_key = f"{history_key}_history"
        st.session_state.clear()
        st.session_state[history_state_key] = [
            {
                "prompt": "What is AI?",
                "response_a": "A: What is AI?",
                "response_b": "B: What is AI?",
            }
        ]
        st.compare_llms(
            model_a="llama-3",
            model_b="gemma-2",
            get_response_a=lambda p: "A: " + p,
            get_response_b=lambda p: "B: " + p,
            history_key=history_key,
        )
        deltas = self.get_all_deltas_from_queue()
        slider_labels = {
            "Evaluate the answer from llama-3:",
            "Evaluate the answer from gemma-2:",
        }
        sliders = [
            d
            for d in deltas
            if hasattr(d, "new_element")
            and hasattr(d.new_element, "slider")
            and getattr(d.new_element.slider, "label", None) in slider_labels
        ]
        self.assertEqual(len(sliders), 2)

    def test_compare_llms_session_state(self):
        """Test that ratings are stored in session state."""
        prompt = "Explain gravity."
        model_a = "modelX"
        model_b = "modelY"
        history_key = "test_session_state"
        self.simulate_chat(prompt, model_a, model_b, history_key=history_key)
        prompt_hash = hashlib.sha256(
            f"{model_a}|{model_b}|{prompt}".encode()
        ).hexdigest()[:8]
        st.session_state[f"rating_{model_a}_{prompt_hash}_a"] = 4
        st.session_state[f"rating_{model_b}_{prompt_hash}_b"] = 2
        self.assertEqual(st.session_state.get(f"rating_{model_a}_{prompt_hash}_a"), 4)
        self.assertEqual(st.session_state.get(f"rating_{model_b}_{prompt_hash}_b"), 2)

    def test_compare_llms_unique_keys(self):
        """Test that different prompts/models generate unique slider keys."""
        history_key1 = "test_unique_keys_1"
        history_key2 = "test_unique_keys_2"
        self.simulate_chat("Prompt 1", "A", "B", history_key=history_key1)
        self.simulate_chat("Prompt 2", "A", "B", history_key=history_key2)
        for prompt in ["Prompt 1", "Prompt 2"]:
            h = hashlib.sha256(f"A|B|{prompt}".encode()).hexdigest()[:8]
            st.session_state[f"rating_A_{h}_a"] = 3
            st.session_state[f"rating_B_{h}_b"] = 3
            self.assertIn(f"rating_A_{h}_a", st.session_state)
            self.assertIn(f"rating_B_{h}_b", st.session_state)

    def test_compare_llms_default_values(self):
        """Test that default slider value is 3 if not set."""
        prompt = "Default test"
        model_a = "A"
        model_b = "B"
        history_key = "test_default_values"
        self.simulate_chat(prompt, model_a, model_b, history_key=history_key)
        h = hashlib.sha256(f"{model_a}|{model_b}|{prompt}".encode()).hexdigest()[:8]
        st.session_state[f"rating_A_{h}_a"] = 3
        st.session_state[f"rating_B_{h}_b"] = 3
        self.assertEqual(st.session_state.get(f"rating_A_{h}_a"), 3)
        self.assertEqual(st.session_state.get(f"rating_B_{h}_b"), 3)

    def test_compare_llms_slider_limits(self):
        """Test that slider values are within the allowed range."""
        prompt = "Test slider limits"
        model_a = "A"
        model_b = "B"
        history_key = "test_slider_limits"
        self.simulate_chat(prompt, model_a, model_b, history_key=history_key)
        h = hashlib.sha256(f"{model_a}|{model_b}|{prompt}".encode()).hexdigest()[:8]
        st.session_state[f"rating_A_{h}_a"] = 1
        st.session_state[f"rating_B_{h}_b"] = 5
        self.assertGreaterEqual(st.session_state.get(f"rating_A_{h}_a"), 1)
        self.assertLessEqual(st.session_state.get(f"rating_A_{h}_a"), 5)
        self.assertGreaterEqual(st.session_state.get(f"rating_B_{h}_b"), 1)
        self.assertLessEqual(st.session_state.get(f"rating_B_{h}_b"), 5)

    def test_compare_llms_different_models(self):
        """Test that sliders are created for different model names."""
        prompt = "Model test"
        model_a = "Model1"
        model_b = "Model2"
        history_key = "test_different_models"
        self.simulate_chat(prompt, model_a, model_b, history_key=history_key)
        h = hashlib.sha256(f"{model_a}|{model_b}|{prompt}".encode()).hexdigest()[:8]
        st.session_state[f"rating_{model_a}_{h}_a"] = 3
        st.session_state[f"rating_{model_b}_{h}_b"] = 3
        self.assertIn(f"rating_{model_a}_{h}_a", st.session_state)
        self.assertIn(f"rating_{model_b}_{h}_b", st.session_state)

    def test_compare_llms_no_key_collision(self):
        """Test that different prompts do not collide in session state."""
        history_key1 = "test_no_collision_1"
        history_key2 = "test_no_collision_2"
        self.simulate_chat("Prompt X", "A", "B", history_key=history_key1)
        self.simulate_chat("Prompt Y", "A", "B", history_key=history_key2)
        h1 = hashlib.sha256(b"A|B|Prompt X").hexdigest()[:8]
        h2 = hashlib.sha256(b"A|B|Prompt Y").hexdigest()[:8]
        st.session_state[f"rating_A_{h1}_a"] = 3
        st.session_state[f"rating_B_{h2}_b"] = 3
        self.assertIn(f"rating_A_{h1}_a", st.session_state)
        self.assertIn(f"rating_B_{h2}_b", st.session_state)

    def test_compare_llms_markdown_rendered(self):
        history_key = "test_markdown_rendered"
        history_state_key = f"{history_key}_history"
        st.session_state.clear()
        st.session_state[history_state_key] = [
            {
                "prompt": "Markdown test",
                "response_a": "**Bold A**",
                "response_b": "*Italic B*",
            }
        ]
        st.compare_llms(
            model_a="A",
            model_b="B",
            get_response_a=lambda p: "**Bold A**",
            get_response_b=lambda p: "*Italic B*",
            history_key=history_key,
        )
        deltas = self.get_all_deltas_from_queue()
        markdowns = [
            d
            for d in deltas
            if hasattr(d, "new_element") and hasattr(d.new_element, "markdown")
        ]
        self.assertTrue(
            any(
                "Bold A" in getattr(d.new_element.markdown, "body", "")
                for d in markdowns
            )
        )
        self.assertTrue(
            any(
                "Italic B" in getattr(d.new_element.markdown, "body", "")
                for d in markdowns
            )
        )

    def test_compare_llms_empty_responses(self):
        history_key = "test_empty_responses"
        history_state_key = f"{history_key}_history"
        st.session_state.clear()
        st.session_state[history_state_key] = [
            {
                "prompt": "Empty response test",
                "response_a": "",
                "response_b": "",
            }
        ]
        st.compare_llms(
            model_a="A",
            model_b="B",
            get_response_a=lambda p: "",
            get_response_b=lambda p: "",
            history_key=history_key,
        )
        deltas = self.get_all_deltas_from_queue()
        slider_labels = {
            "Evaluate the answer from A:",
            "Evaluate the answer from B:",
        }
        sliders = [
            d
            for d in deltas
            if hasattr(d, "new_element")
            and hasattr(d.new_element, "slider")
            and getattr(d.new_element.slider, "label", None) in slider_labels
        ]
        self.assertEqual(len(sliders), 2)

    def test_compare_llms_same_model_names(self):
        """Test that sliders have unique keys even if model names are the same."""
        history_key = "test_same_model_names"
        self.simulate_chat("Same model test", "A", "A", history_key=history_key)
        h = hashlib.sha256(b"A|A|Same model test").hexdigest()[:8]
        st.session_state[f"rating_A_{h}_a"] = 3
        st.session_state[f"rating_A_{h}_b"] = 3
        self.assertIn(f"rating_A_{h}_a", st.session_state)
        self.assertIn(f"rating_A_{h}_b", st.session_state)
        self.assertNotEqual(f"rating_A_{h}_a", f"rating_A_{h}_b")

    def test_compare_llms_long_prompt(self):
        """Test that a very long prompt does not break the widget."""
        long_prompt = "x" * 1000
        history_key = "test_long_prompt"
        self.simulate_chat(long_prompt, "A", "B", history_key=history_key)
        h = hashlib.sha256(f"A|B|{long_prompt}".encode()).hexdigest()[:8]
        st.session_state[f"rating_A_{h}_a"] = 3
        st.session_state[f"rating_B_{h}_b"] = 3
        self.assertIn(f"rating_A_{h}_a", st.session_state)
        self.assertIn(f"rating_B_{h}_b", st.session_state)

    def test_compare_llms_ratings_independent(self):
        """Test that ratings for different prompts are independent."""
        prompt1 = "Prompt 1"
        prompt2 = "Prompt 2"
        history_key1 = "test_ratings_independent_1"
        history_key2 = "test_ratings_independent_2"
        self.simulate_chat(prompt1, "A", "B", history_key=history_key1)
        h1 = hashlib.sha256(b"A|B|Prompt 1").hexdigest()[:8]
        st.session_state[f"rating_A_{h1}_a"] = 2
        st.session_state[f"rating_B_{h1}_b"] = 5

        self.simulate_chat(prompt2, "A", "B", history_key=history_key2)
        h2 = hashlib.sha256(b"A|B|Prompt 2").hexdigest()[:8]
        st.session_state[f"rating_A_{h2}_a"] = 4
        st.session_state[f"rating_B_{h2}_b"] = 1

        self.assertEqual(st.session_state.get(f"rating_A_{h1}_a"), 2)
        self.assertEqual(st.session_state.get(f"rating_B_{h1}_b"), 5)
        self.assertEqual(st.session_state.get(f"rating_A_{h2}_a"), 4)
        self.assertEqual(st.session_state.get(f"rating_B_{h2}_b"), 1)
