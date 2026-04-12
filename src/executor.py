"""
A2A Executor — handles incoming tasks from the MLE-Bench green agent.
Receives competition tar.gz, runs ML agent, returns submission.csv.
"""
import asyncio
import base64
import logging
import os
import tarfile
import tempfile

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.types import (
    Artifact,
    FilePart,
    FileWithBytes,
    Message,
    Part,
    TaskArtifactUpdateEvent,
    TaskState,
    TaskStatus,
    TaskStatusUpdateEvent,
    TextPart,
    UnsupportedOperationError,
)
from a2a.utils import new_task

from ml_agent import run_ml_agent

logger = logging.getLogger(__name__)


def _make_status(state: TaskState, text: str) -> TaskStatus:
    return TaskStatus(
        state=state,
        message=Message(
            role="agent",
            parts=[Part(root=TextPart(text=text))],
        ),
    )


class Executor(AgentExecutor):

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        task = context.current_task or new_task(context.message)

        await event_queue.enqueue_event(
            TaskStatusUpdateEvent(
                task_id=task.id,
                context_id=task.context_id,
                status=_make_status(TaskState.working, "Starting ML agent..."),
                final=False,
            )
        )

        try:
            # Extract tar.gz and instructions from the message
            workdir, instructions = await asyncio.get_event_loop().run_in_executor(
                None, self._extract_input, context.message
            )

            loop = asyncio.get_event_loop()

            def on_status(msg: str):
                asyncio.run_coroutine_threadsafe(
                    event_queue.enqueue_event(
                        TaskStatusUpdateEvent(
                            task_id=task.id,
                            context_id=task.context_id,
                            status=_make_status(TaskState.working, msg),
                            final=False,
                        )
                    ),
                    loop,
                )

            # Run the ML agent in a thread (it's blocking)
            submission_path = await asyncio.get_event_loop().run_in_executor(
                None, lambda: run_ml_agent(workdir, instructions, on_status=on_status)
            )

            # Read and encode submission.csv
            with open(submission_path, "rb") as f:
                csv_bytes = f.read()
            csv_b64 = base64.b64encode(csv_bytes).decode()

            artifact = Artifact(
                name="submission.csv",
                parts=[
                    Part(
                        root=FilePart(
                            file=FileWithBytes(
                                name="submission.csv",
                                mime_type="text/csv",
                                bytes=csv_b64,
                            )
                        )
                    )
                ],
            )

            await event_queue.enqueue_event(
                TaskArtifactUpdateEvent(
                    task_id=task.id,
                    context_id=task.context_id,
                    artifact=artifact,
                    append=False,
                    last_chunk=True,
                )
            )

            await event_queue.enqueue_event(
                TaskStatusUpdateEvent(
                    task_id=task.id,
                    context_id=task.context_id,
                    status=_make_status(TaskState.completed, "submission.csv ready"),
                    final=True,
                )
            )

        except Exception as e:
            logger.exception("Agent failed")
            await event_queue.enqueue_event(
                TaskStatusUpdateEvent(
                    task_id=task.id,
                    context_id=task.context_id,
                    status=_make_status(TaskState.failed, f"Error: {e}"),
                    final=True,
                )
            )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise UnsupportedOperationError()

    # ------------------------------------------------------------------ helpers

    def _extract_input(self, message: Message) -> tuple[str, str]:
        """Extract tar.gz archive and text instructions from A2A message parts."""
        workdir = tempfile.mkdtemp(prefix="mle_agent_")
        instructions = ""
        archive_extracted = False

        for part in message.parts:
            p = part.root if hasattr(part, "root") else part

            if isinstance(p, TextPart):
                instructions += p.text + "\n"

            elif isinstance(p, FilePart):
                file_data = p.file
                raw = None
                if hasattr(file_data, "bytes") and file_data.bytes:
                    raw = base64.b64decode(file_data.bytes)
                elif hasattr(file_data, "uri") and file_data.uri:
                    import urllib.request
                    with urllib.request.urlopen(file_data.uri) as resp:
                        raw = resp.read()

                if raw:
                    archive_path = os.path.join(workdir, "competition.tar.gz")
                    with open(archive_path, "wb") as f:
                        f.write(raw)
                    with tarfile.open(archive_path, "r:gz") as tar:
                        tar.extractall(workdir)
                    archive_extracted = True
                    logger.info(f"Extracted competition archive to {workdir}")

        if not archive_extracted:
            logger.warning("No tar.gz archive found in message parts")

        return workdir, instructions.strip()
