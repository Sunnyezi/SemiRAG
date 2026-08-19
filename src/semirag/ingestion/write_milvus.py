"""Import a directory of Markdown files into the configured Milvus collection."""

import multiprocessing
from multiprocessing import Queue
from pathlib import Path

from semirag.config import KNOWLEDGE_BASE_DIR
from semirag.ingestion.markdown_parser import MarkdownParser
from semirag.ingestion.milvus_db import MilvusVectorSave
from semirag.utils.log_utils import log


def file_parser_process(dir_path: str | Path, output_queue: Queue, batch_size: int = 20) -> None:
    """Parse top-level Markdown files and send document batches to the writer."""
    source_dir = Path(dir_path)
    log.info(f"解析进程开始扫描目录: {source_dir}")
    md_files = sorted(source_dir.glob("*.md"))

    if not md_files:
        log.warning("警告：未找到任何 Markdown 文件")
        output_queue.put(None)
        return

    parser = MarkdownParser()
    doc_batch = []
    for file_path in md_files:
        try:
            documents = parser.parse_markdown_to_documents(str(file_path))
            if documents:
                doc_batch.extend(documents)
            if len(doc_batch) >= batch_size:
                output_queue.put(doc_batch.copy())
                doc_batch.clear()
        except Exception as error:
            log.error(f"解析失败 {file_path}: {error}")
            log.exception(error)

    if doc_batch:
        output_queue.put(doc_batch)
    output_queue.put(None)
    log.info(f"解析完成，共处理 {len(md_files)} 个文件")


def milvus_writer_process(input_queue: Queue) -> None:
    """Read document batches from the queue and persist them to Milvus."""
    log.info("Milvus 写入进程启动")
    vector_store = MilvusVectorSave()
    vector_store.create_connection()
    total_count = 0

    while True:
        try:
            documents = input_queue.get()
            if documents is None:
                break
            if isinstance(documents, list):
                vector_store.add_documents(documents)
                total_count += len(documents)
                log.info(f"累计已写入: {total_count} 个文档")
        except Exception as error:
            log.error("Milvus 写入失败")
            log.exception(error)

    log.info(f"写入进程结束，总计写入 {total_count} 个文档")


def ingest_directory(md_dir: str | Path = KNOWLEDGE_BASE_DIR, queue_maxsize: int = 20) -> None:
    """Rebuild the configured Milvus collection from a Markdown directory."""
    source_dir = Path(md_dir)
    if not source_dir.is_dir():
        raise FileNotFoundError(f"知识库目录不存在: {source_dir}")

    # This intentionally recreates the configured collection before importing.
    vector_store = MilvusVectorSave()
    vector_store.create_collection()

    docs_queue = Queue(maxsize=queue_maxsize)
    parser_process = multiprocessing.Process(
        target=file_parser_process,
        args=(source_dir, docs_queue),
    )
    writer_process = multiprocessing.Process(
        target=milvus_writer_process,
        args=(docs_queue,),
    )

    parser_process.start()
    writer_process.start()
    parser_process.join()
    writer_process.join()
    print("系统提示：所有任务完成")


def main() -> None:
    ingest_directory()


if __name__ == "__main__":
    main()
