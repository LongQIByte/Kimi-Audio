#!/usr/bin/env python3
import os
import gradio as gr
import soundfile as sf
import time
import tempfile
import logging
from datetime import datetime
from kimia_infer.api.kimia import KimiAudio

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class KimiWebChat:
    def __init__(
        self,
        model_path="moonshotai/Kimi-Audio-7B-Instruct",
        output_dir="output",
    ):
        """Initialize Kimi Audio Web Chat"""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Model configuration
        logger.info(f"Loading model: {model_path}")
        self.model = KimiAudio(model_path=model_path, load_detokenizer=True)
        logger.info("Model loaded successfully!")

        # Default sampling parameters
        self.sampling_params = {
            "audio_temperature": 0.8,
            "audio_top_k": 10,
            "text_temperature": 0.0,
            "text_top_k": 5,
            "audio_repetition_penalty": 1.0,
            "audio_repetition_window_size": 64,
            "text_repetition_penalty": 1.0,
            "text_repetition_window_size": 16,
        }

        # Audio sample rate
        self.sample_rate = 24000

        # Chat history
        self.chat_history = []

    def process_chat(self, user_input, audio_input, history):
        """Process chat request"""
        try:
            # Build message list
            messages = []

            # Add historical messages
            for h in history:
                if h[0]:  # User message
                    if isinstance(h[0], list):
                        messages.append(
                            {"role": "user", "message_type": "text", "content": h[0][0]}
                        )
                    else:
                        messages.append(
                            {
                                "role": "user",
                                "message_type": "audio",
                                "content": h[0][1],
                            }
                        )
                if h[1]:  # Assistant message
                    messages.append(
                        {
                            "role": "assistant",
                            "message_type": "audio-text",
                            "content": h[1],
                        }
                    )

            # Process current user input
            audio_path = None  # Initialize audio path

            if user_input:
                messages.append(
                    {"role": "user", "message_type": "text", "content": user_input}
                )

            if audio_input is not None:
                # Save audio file
                timestamp = int(time.time())
                audio_path = os.path.join(self.output_dir, f"input_{timestamp}.wav")
                sf.write(audio_path, audio_input[1], audio_input[0])
                messages.append(
                    {"role": "user", "message_type": "audio", "content": audio_path}
                )
                logger.info(f"Saved user audio: {audio_path}")

            if not user_input and audio_input is None:
                return history, history, None, ""

            # Generate response
            logger.info("Generating response...")
            wav_output, text_output = self.model.generate(
                messages, **self.sampling_params, output_type="both"
            )

            # Process text response
            text_reply = (
                text_output
                if text_output
                else "Sorry, I didn't generate a text response."
            )

            # Process audio response
            audio_response = None
            output_path = None
            if wav_output is not None:
                timestamp = int(time.time())
                output_path = os.path.join(self.output_dir, f"output_{timestamp}.wav")
                audio_data = wav_output.detach().cpu().view(-1).numpy()
                sf.write(output_path, audio_data, self.sample_rate)
                audio_response = output_path
                logger.info(f"Saved AI audio response: {output_path}")

            # Construct display message: if there's audio, show "🎵 [Audio Response] + text", otherwise just text
            if output_path:
                bot_display_message = f"🎵 [Audio Response]\n{text_reply}"
            else:
                bot_display_message = text_reply

            # Construct history message: format for next conversation [audio_path, text_content]
            bot_history_message = (
                [output_path, text_reply] if output_path else text_reply
            )

            # Construct user message in different formats
            # User message for display (Chatbot interface)
            if user_input and audio_path:
                user_display_msg = f"{user_input}\n🎤 [Contains Audio]"
            elif user_input:
                user_display_msg = user_input
            else:
                user_display_msg = "🎤 [Audio Message]"

            # User message for model processing (history format)
            if user_input and audio_path:
                user_history_msg = [user_input, audio_path]
            elif user_input:
                user_history_msg = user_input
            else:
                user_history_msg = audio_path

            # History for display (Chatbot display)
            display_history = history + [[user_display_msg, bot_display_message]]

            # History for model processing (contains complete information)
            model_history = history + [[user_history_msg, bot_history_message]]

            return display_history, model_history, audio_response, ""

        except Exception as e:
            logger.error(f"Error processing chat: {str(e)}")
            error_msg = f"Processing error: {str(e)}"
            new_history = history + [[user_input or "[Audio Message]", error_msg]]
            return new_history, new_history, None, ""

    def clear_chat(self):
        """Clear chat history"""
        logger.info("Clearing chat history")
        return [], [], None, ""

    def update_params(
        self,
        audio_temp,
        audio_topk,
        text_temp,
        text_topk,
        audio_rep_penalty,
        text_rep_penalty,
    ):
        """Update model parameters"""
        self.sampling_params.update(
            {
                "audio_temperature": audio_temp,
                "audio_top_k": int(audio_topk),
                "text_temperature": text_temp,
                "text_top_k": int(text_topk),
                "audio_repetition_penalty": audio_rep_penalty,
                "text_repetition_penalty": text_rep_penalty,
            }
        )
        logger.info(f"Parameters updated: {self.sampling_params}")
        return "✅ Parameters updated successfully!"


def create_demo(args):
    """Create Gradio interface"""

    # Initialize chat handler
    chat_handler = KimiWebChat(args.model_path)

    # Custom CSS
    css = """
    .chatbot {
        height: 500px;
    }
    .audio-container {
        margin: 10px 0;
    }
    """

    with gr.Blocks(css=css, title="Kimi-Audio Voice Chatbot") as demo:
        gr.Markdown("# 🎤 Kimi-Audio Voice Assistant")
        gr.Markdown(
            "Supports text and voice input, AI responds with both text and voice messages"
        )

        with gr.Tab("💬 Chat"):
            with gr.Row():
                with gr.Column(scale=3):
                    # Chat display area
                    chatbot = gr.Chatbot(
                        label="Conversation History", height=500, show_label=True
                    )

                    # User input area
                    with gr.Row():
                        with gr.Column(scale=4):
                            user_input = gr.Textbox(
                                placeholder="Enter your message...",
                                label="Text Input",
                                lines=2,
                            )
                        with gr.Column(scale=1):
                            send_btn = gr.Button("Send", variant="primary", size="lg")

                    # Audio input
                    audio_input = gr.Audio(
                        label="Voice Input (Record or upload audio file)", type="numpy"
                    )

                    # Control buttons
                    with gr.Row():
                        clear_btn = gr.Button("Clear Chat", variant="secondary")

                with gr.Column(scale=1):
                    # AI audio response
                    audio_output = gr.Audio(
                        label="🤖 AI Voice Response", interactive=False, autoplay=True
                    )

                    # Quick tips
                    gr.Markdown("### 💡 Usage Tips")
                    gr.Markdown(
                        """
                    - You can input text or record voice
                    - AI will respond with both text and voice
                    - Each conversation preserves history
                    - You can clear the conversation to start fresh
                    """
                    )

        with gr.Tab("⚙️ Settings"):
            gr.Markdown("### Model Parameter Adjustment")

            with gr.Row():
                with gr.Column():
                    gr.Markdown("**Audio Generation Parameters**")
                    audio_temp = gr.Slider(
                        0.0,
                        2.0,
                        value=0.8,
                        step=0.1,
                        label="Audio Temperature",
                        info="Controls randomness in audio generation",
                    )
                    audio_topk = gr.Slider(
                        1,
                        50,
                        value=10,
                        step=1,
                        label="Audio Top-K",
                        info="Controls number of candidate audio tokens",
                    )
                    audio_rep_penalty = gr.Slider(
                        0.1,
                        3.0,
                        value=1.0,
                        step=0.1,
                        label="Audio Repetition Penalty",
                        info="Prevents repetitive audio segments",
                    )

                with gr.Column():
                    gr.Markdown("**Text Generation Parameters**")
                    text_temp = gr.Slider(
                        0.0,
                        2.0,
                        value=0.0,
                        step=0.1,
                        label="Text Temperature",
                        info="Controls randomness in text generation",
                    )
                    text_topk = gr.Slider(
                        1,
                        50,
                        value=5,
                        step=1,
                        label="Text Top-K",
                        info="Controls number of candidate text tokens",
                    )
                    text_rep_penalty = gr.Slider(
                        0.1,
                        3.0,
                        value=1.0,
                        step=0.1,
                        label="Text Repetition Penalty",
                        info="Prevents repetitive text",
                    )

            update_btn = gr.Button("Apply Settings", variant="primary")
            param_status = gr.Textbox(label="Settings Status", interactive=False)

        with gr.Tab("ℹ️ About"):
            gr.Markdown(
                """
            # Kimi-Audio Multimodal Chatbot
            
            ## 🌟 Features
            - **Multimodal Interaction**: Supports text and voice input/output
            - **Real-time Conversation**: Fast response, natural communication
            - **History Recording**: Maintains conversation continuity
            - **Parameter Adjustment**: Customizable generation parameters for optimal experience
            
            ## 🚀 How to Use
            1. In the "Chat" tab, input text or record voice
            2. AI will generate both text and voice responses
            3. Adjust model parameters in the "Settings" tab
            4. Use "Clear Chat" to start a new conversation
            
            ## 📝 Technical Information
            - Based on Kimi-Audio-7B-Instruct model
            - Supports Chinese and English conversations
            - Audio sample rate: 24kHz
            
            ---
            
            🔗 [Project Homepage](https://github.com/moonshotai/Kimi-Audio)
            """
            )

        # Hidden state variable for saving history
        history_state = gr.State([])

        # Event binding
        def handle_chat(user_text, audio, history):
            return chat_handler.process_chat(user_text, audio, history)

        # Send button event
        send_btn.click(
            fn=handle_chat,
            inputs=[user_input, audio_input, history_state],
            outputs=[chatbot, history_state, audio_output, user_input],
        )

        # Enter to send
        user_input.submit(
            fn=handle_chat,
            inputs=[user_input, audio_input, history_state],
            outputs=[chatbot, history_state, audio_output, user_input],
        )

        # Clear conversation
        clear_btn.click(
            fn=chat_handler.clear_chat,
            outputs=[chatbot, history_state, audio_output, user_input],
        )

        # Update parameters
        update_btn.click(
            fn=chat_handler.update_params,
            inputs=[
                audio_temp,
                audio_topk,
                text_temp,
                text_topk,
                audio_rep_penalty,
                text_rep_penalty,
            ],
            outputs=param_status,
        )

    return demo


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Kimi-Audio Web Demo")
    parser.add_argument(
        "--model_path",
        type=str,
        default="moonshotai/Kimi-Audio-7B-Instruct",
        help="Model path",
    )
    parser.add_argument("--port", type=int, default=7860, help="Port number")
    parser.add_argument("--share", action="store_true", help="Create public link")

    args = parser.parse_args()

    logger.info(f"Starting Kimi-Audio Web Demo")
    logger.info(f"Model path: {args.model_path}")
    logger.info(f"Port: {args.port}")

    demo = create_demo(args)
    demo.launch(
        server_port=args.port, server_name="0.0.0.0", share=args.share, show_api=False
    )
