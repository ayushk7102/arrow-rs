use arrow_array::{RecordBatch, RecordBatchReader};
use arrow_pyarrow::PyArrowType;
use arrow_schema::{ArrowError, Schema, SchemaRef};
use futures::TryStreamExt;
use parquet::arrow::ProjectionMask;
use parquet::arrow::arrow_reader::ArrowReaderOptions;
use parquet::arrow::async_reader::ParquetRecordBatchStreamBuilder;
use parquet::file::metadata::PageIndexPolicy;
use pyo3::exceptions::{PyIOError, PyRuntimeError};
use pyo3::prelude::*;
use std::sync::Arc;
use std::sync::mpsc::{self, Receiver};
use tokio::runtime::Runtime;

struct SyncParquetReader {
    schema: SchemaRef,
    rx: Receiver<Result<RecordBatch, ArrowError>>,
    _runtime: Arc<Runtime>,
}

impl Iterator for SyncParquetReader {
    type Item = Result<RecordBatch, ArrowError>;
    fn next(&mut self) -> Option<Self::Item> {
        self.rx.recv().ok()
    }
}

impl RecordBatchReader for SyncParquetReader {
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }
}

#[pyfunction]
fn read_parquet_schema(path: &str) -> PyResult<PyArrowType<Schema>> {
    let rt = Runtime::new().map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let schema = rt.block_on(async {
        let file = tokio::fs::File::open(path)
            .await
            .map_err(|e| PyIOError::new_err(e.to_string()))?;
        let builder = ParquetRecordBatchStreamBuilder::new(file)
            .await
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok::<_, PyErr>(builder.schema().as_ref().clone())
    })?;
    Ok(PyArrowType(schema))
}

#[pyfunction]
#[pyo3(signature = (path, columns=None, row_groups=None, batch_size=65536))]
fn read_parquet_batches_stream(
    path: &str,
    columns: Option<Vec<String>>,
    row_groups: Option<Vec<usize>>,
    batch_size: usize,
) -> PyResult<PyArrowType<Box<dyn RecordBatchReader + Send>>> {
    let rt = Arc::new(
        tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
    );

    let path = path.to_string();
    let (tx, rx) = mpsc::channel::<Result<RecordBatch, ArrowError>>();

    let schema: SchemaRef = rt.block_on(async {
        let file = tokio::fs::File::open(&path)
            .await
            .map_err(|e| PyIOError::new_err(e.to_string()))?;

        let options =
            ArrowReaderOptions::new().with_page_index_policy(PageIndexPolicy::default());
        let mut builder = ParquetRecordBatchStreamBuilder::new_with_options(file, options)
            .await
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
            .with_batch_size(batch_size);

        if let Some(ref rgs) = row_groups {
            builder = builder.with_row_groups(rgs.clone());
        }

        if let Some(ref cols) = columns {
            let pq_schema = builder.parquet_schema().clone();
            let indices: Vec<usize> = cols
                .iter()
                .filter_map(|c| {
                    (0..pq_schema.num_columns())
                        .find(|&i| pq_schema.column(i).name() == c)
                })
                .collect();
            let mask = ProjectionMask::roots(builder.parquet_schema(), indices);
            builder = builder.with_projection(mask);
        }

        let schema = builder.schema().clone();

        let mut stream = builder
            .build()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        let tx2 = tx.clone();
        tokio::spawn(async move {
            while let Ok(Some(batch)) = stream.try_next().await {
                if tx2.send(Ok(batch)).is_err() {
                    break;
                }
            }
        });

        Ok::<SchemaRef, PyErr>(schema)
    })?;

    let reader = SyncParquetReader {
        schema,
        rx,
        _runtime: rt,
    };

    Ok(PyArrowType(
        Box::new(reader) as Box<dyn RecordBatchReader + Send>
    ))
}

#[pymodule]
fn ray_parquet_rs(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(read_parquet_schema, m)?)?;
    m.add_function(wrap_pyfunction!(read_parquet_batches_stream, m)?)?;
    Ok(())
}
