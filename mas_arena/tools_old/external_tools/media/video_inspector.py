from typing import Optional, Dict, Any, List

from smolagents import Tool
from smolagents.models import Model


class VideoInspectorTool(Tool):
    name = "inspect_file_as_video"
    description = """
You cannot load files directly: use this tool to process video files and answer related questions.
This tool supports the following video formats: [".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"]. For other file types, use the appropriate inspection tool."""

    inputs = {
        "file_path": {
            "description": "The path to the file you want to read as video. Must be a video file with extensions like '.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv'. If it is text, use the text_inspector tool instead! If it is an image, use the visual_inspector tool instead! If it is audio, use the audio_inspector tool instead!",
            "type": "string",
        },
        "question": {
            "description": "[Optional]: Your question about the video content. Provide as much context as possible. Do not pass this parameter if you just want to directly return the content analysis of the file.",
            "type": "string",
            "nullable": True,
        },
        "sample_rate": {
            "description": "[Optional]: Number of frames to sample per second for analysis (default: 1)",
            "type": "number",
            "nullable": True,
        },
        "start_time": {
            "description": "[Optional]: Start time of the video segment in seconds (default: 0)",
            "type": "number",
            "nullable": True,
        },
        "end_time": {
            "description": "[Optional]: End time of the video segment in seconds (default: None for full video)",
            "type": "number",
            "nullable": True,
        },
    }
    output_type = "string"

    def __init__(self, model: Model = None, text_limit: int = 1000):
        # Local imports to satisfy tool validators in sandboxed execution
        from dotenv import load_dotenv

        load_dotenv()
        super().__init__()
        self.model = model
        self.text_limit = text_limit
        self.supported_extensions = (".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv")

    def _validate_file_type(self, file_path: str):
        """Validate if the file type is a supported video format"""
        # NOTE: keep this implementation extremely simple because some tool validators
        # may not understand generator/comprehension local variables (e.g. `ext`).
        if not file_path.lower().endswith(self.supported_extensions):
            raise ValueError(
                f"Unsupported file type. Supported video formats: {list(self.supported_extensions)}. "
                "Use the appropriate tool for text/image/audio files."
            )

    def extract_metadata(self, file_path: str) -> Dict[str, Any]:
        """Extract metadata from video file using OpenCV"""
        from typing import Dict, Any
        import os
        from datetime import timedelta

        import cv2
        try:
            cap = cv2.VideoCapture(file_path)
            if not cap.isOpened():
                return {"error": "Unable to open video file or unsupported format"}

            metadata = {}

            # Basic file information
            metadata["file_size"] = f"{os.path.getsize(file_path) / (1024*1024):.2f} MB"

            # Video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            metadata["fps"] = f"{fps:.2f}"
            metadata["total_frames"] = frame_count
            metadata["resolution"] = f"{width}x{height}"
            metadata["aspect_ratio"] = (
                f"{width/height:.2f}" if height > 0 else "Unknown"
            )

            # Duration
            if fps > 0:
                duration_seconds = frame_count / fps
                duration = timedelta(seconds=int(duration_seconds))
                metadata["duration"] = str(duration)
                metadata["duration_seconds"] = f"{duration_seconds:.2f}"
            else:
                metadata["duration"] = "Unknown"
                metadata["duration_seconds"] = "Unknown"

            # Additional codec information
            fourcc = cap.get(cv2.CAP_PROP_FOURCC)
            codec = "".join([chr((int(fourcc) >> 8 * i) & 0xFF) for i in range(4)])
            metadata["codec"] = codec if codec.strip() else "Unknown"

            cap.release()
            return metadata

        except Exception as e:
            return {"error": f"Metadata extraction failed: {str(e)}"}

    def extract_frames(
        self,
        file_path: str,
        sample_rate: float = 1.0,
        start_time: float = 0.0,
        end_time: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Extract frames from video with given sample rate"""
        from typing import Optional, List, Dict, Any
        import base64

        import cv2
        try:
            cap = cv2.VideoCapture(file_path)
            if not cap.isOpened():
                raise ValueError("Unable to open video file")

            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if fps <= 0:
                raise ValueError("Invalid video FPS")

            # Calculate frame range
            start_frame = int(start_time * fps)
            if end_time is not None:
                end_frame = min(int(end_time * fps), total_frames)
            else:
                end_frame = total_frames

            # Calculate sampling interval
            frames_per_sample = max(1, int(fps / sample_rate))

            frames = []
            current_frame = start_frame

            while current_frame < end_frame:
                cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
                ret, frame = cap.read()

                if not ret:
                    break

                # Convert frame to base64 for LLM consumption
                _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                frame_base64 = base64.b64encode(buffer.tobytes()).decode("utf-8")

                timestamp = current_frame / fps
                frames.append(
                    {
                        "data": f"data:image/jpeg;base64,{frame_base64}",
                        "timestamp": timestamp,
                        "frame_number": current_frame,
                    }
                )

                current_frame += frames_per_sample

                # Limit total frames to prevent excessive token usage
                if len(frames) >= 50:  # Reasonable limit for LLM processing
                    break

            cap.release()
            return frames

        except Exception as e:
            raise RuntimeError(f"Frame extraction failed: {str(e)}") from e

    def create_video_content(
        self, prompt: str, frames: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Create content for LLM consumption with frames"""
        from typing import List, Dict, Any

        content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]

        for frame in frames:
            content.append({"type": "image_url", "image_url": {"url": frame["data"]}})

        return content

    def forward(
        self,
        file_path: str,
        question: Optional[str] = None,
        sample_rate: Optional[float] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> str:
        from typing import Optional
        from smolagents.models import MessageRole

        # Set defaults
        sample_rate = sample_rate or 1.0
        start_time = start_time or 0.0

        # Validate file type
        self._validate_file_type(file_path)

        # Extract video metadata
        metadata = self.extract_metadata(file_path)

        # Format metadata
        metadata_str = "Video Metadata:\n"
        for key, value in metadata.items():
            metadata_str += f"  {key.replace('_', ' ').title()}: {value}\n"
        metadata_str += "\n"

        # Extract frames for analysis
        try:
            frames = self.extract_frames(file_path, sample_rate, start_time, end_time)
            if not frames:
                return f"Video processing error: No frames could be extracted\n\n{metadata_str}"

            frames_info = f"Extracted {len(frames)} frames for analysis "
            frames_info += f"(sample rate: {sample_rate} fps, "
            frames_info += f"time range: {start_time:.1f}s to {end_time or 'end'}s)\n\n"

        except Exception as e:
            return f"Video processing error: {str(e)}\n\n{metadata_str}"

        if not question:
            # Return basic analysis without specific question
            basic_prompt = (
                "Analyze this video content. Describe what you see including:\n"
                "1. Main subjects and objects\n"
                "2. Setting and environment\n"
                "3. Actions and movements\n"
                "4. Key visual elements\n"
                "5. Overall narrative or theme\n\n"
                f"Video contains {len(frames)} frames from {frames[0]['timestamp']:.1f}s to {frames[-1]['timestamp']:.1f}s"
            )

            messages = [
                {
                    "role": MessageRole.SYSTEM,
                    "content": self.create_video_content(
                        f"{metadata_str}{frames_info}{basic_prompt}", frames
                    ),
                },
                {
                    "role": MessageRole.USER,
                    "content": [
                        {"type": "text", "text": "Please analyze this video content."}
                    ],
                },
            ]
        else:
            # Answer specific question about video
            specific_prompt = (
                f"Analyze this video to answer the following question: {question}\n\n"
                "Please provide:\n"
                "1. Direct answer to the question\n"
                "2. Supporting visual evidence from the video\n"
                "3. Relevant context and details\n"
                "4. Confidence level in your answer\n\n"
                f"Video contains {len(frames)} frames from {frames[0]['timestamp']:.1f}s to {frames[-1]['timestamp']:.1f}s"
            )

            messages = [
                {
                    "role": MessageRole.SYSTEM,
                    "content": self.create_video_content(
                        f"{metadata_str}{frames_info}{specific_prompt}", frames
                    ),
                },
                {
                    "role": MessageRole.USER,
                    "content": [{"type": "text", "text": f"Question: {question}"}],
                },
            ]

        try:
            content = self.model(messages).content
            if not isinstance(content, str):
                content = str(content)
            return content
        except Exception as e:
            return f"Analysis failed: {str(e)}\n\n{metadata_str}{frames_info}"


if __name__ == "__main__":
    model = Model(model_name="gpt-4o-mini")
    tool = VideoInspectorTool(model=model, text_limit=1000)
    print(tool.forward(file_path="/workspace/project_memory_layer/download_files/video.mp4", question="What is the video about?"))
