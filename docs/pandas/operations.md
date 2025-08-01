# Semantic Operations

The pandas integration provides several semantic operations through the `.semantic` accessor. Each operation is designed to handle specific types of transformations and analyses using LLMs.

All semantic operations return a new DataFrame that preserves the original columns and adds new columns based on the output schema. For example, if your original DataFrame has a column `text` and you use `map` with an `output={"schema": {"sentiment": "str", "keywords": "list[str]"}}`, the resulting DataFrame will have three columns: `text`, `sentiment`, and `keywords`. This makes it easy to chain operations and maintain data lineage.

## Map Operation

::: docetl.apis.pd_accessors.SemanticAccessor.map
    options:
        show_root_heading: false
        heading_level: 3

Example usage:
```python
# Basic map operation
df.semantic.map(
    prompt="Extract sentiment and key points from: {{input.text}}",
    output={
        "schema": {
            "sentiment": "str",
            "key_points": "list[str]"
        }
    },
    validate=["len(output['key_points']) <= 5"],
    num_retries_on_validate_failure=2
)

# Using structured output mode for better JSON schema support
df.semantic.map(
    prompt="Extract detailed information from: {{input.text}}",
    output={
        "schema": {
            "company": "str",
            "product": "str",
            "features": "list[str]"
        },
        "mode": "structured_output"
    }
)

# Backward compatible syntax (still supported)
df.semantic.map(
    prompt="Extract sentiment from: {{input.text}}",
    output_schema={"sentiment": "str"}
)
```

## Filter Operation

::: docetl.apis.pd_accessors.SemanticAccessor.filter
    options:
        show_root_heading: false
        heading_level: 3

Example usage:
```python
# Simple filtering
df.semantic.filter(
    prompt="Is this text about technology? {{input.text}}"
)

# Custom output schema with reasons
df.semantic.filter(
    prompt="Analyze if this is relevant: {{input.text}}",
    output={
        "schema": {
            "keep": "bool",
            "reason": "str"
        }
    }
)
```

## Merge Operation (Experimental)

> **Note**: The merge operation is an experimental feature based on our equijoin operator. It provides a pandas-like interface for semantic record matching and deduplication. When `fuzzy=True`, it automatically invokes optimization to improve performance while maintaining accuracy.

::: docetl.apis.pd_accessors.SemanticAccessor.merge
    options:
        show_root_heading: false
        heading_level: 3

Example usage:
```python
# Simple merge
merged_df = df1.semantic.merge(
    df2,
    comparison_prompt="Are these records about the same entity? {{input1}} vs {{input2}}"
)

# Fuzzy merge with optimization
merged_df = df1.semantic.merge(
    df2,
    comparison_prompt="Compare: {{input1}} vs {{input2}}",
    fuzzy=True,
    target_recall=0.9
)
```

## Aggregate Operation

::: docetl.apis.pd_accessors.SemanticAccessor.agg
    options:
        show_root_heading: false
        heading_level: 3

Example usage:
```python
# Simple aggregation
df.semantic.agg(
    reduce_prompt="Summarize these items: {{input.text}}",
    output_schema={"summary": "str"}
)

# Fuzzy matching with custom resolution
df.semantic.agg(
    reduce_prompt="Combine these items: {{input.text}}",
    output_schema={"combined": "str"},
    fuzzy=True,
    comparison_prompt="Are these items similar: {{input1.text}} vs {{input2.text}}",
    resolution_prompt="Resolve conflicts between: {{items}}",
    resolution_output_schema={"resolved": "str"}
)
```

## Split Operation

::: docetl.apis.pd_accessors.SemanticAccessor.split
    options:
        show_root_heading: false
        heading_level: 3

Example usage:
```python
# Split by token count
df.semantic.split(
    split_key="content",
    method="token_count",
    method_kwargs={"num_tokens": 100}
)

# Split by delimiter
df.semantic.split(
    split_key="text",
    method="delimiter",
    method_kwargs={"delimiter": "\n\n", "num_splits_to_group": 2}
)
```

## Gather Operation

::: docetl.apis.pd_accessors.SemanticAccessor.gather
    options:
        show_root_heading: false
        heading_level: 3

Example usage:
```python
# Basic gathering with surrounding context
df.semantic.gather(
    content_key="chunk_content",
    doc_id_key="document_id",
    order_key="chunk_number",
    peripheral_chunks={
        "previous": {"head": {"count": 2}, "tail": {"count": 1}},
        "next": {"head": {"count": 1}, "tail": {"count": 2}}
    }
)

# Simple gathering without peripheral chunks
df.semantic.gather(
    content_key="content",
    doc_id_key="doc_id",
    order_key="order"
)
```

## Unnest Operation

::: docetl.apis.pd_accessors.SemanticAccessor.unnest
    options:
        show_root_heading: false
        heading_level: 3

Example usage:
```python
# Unnest a list column
df.semantic.unnest(unnest_key="tags")

# Unnest a dictionary column with specific fields
df.semantic.unnest(
    unnest_key="user_info",
    expand_fields=["name", "age"]
)

# Recursive unnesting with depth control
df.semantic.unnest(
    unnest_key="nested_lists",
    recursive=True,
    depth=2
)
```

## Common Features

All operations support:

1. **Cost Tracking**
```python
# After any operation
print(f"Operation cost: ${df.semantic.total_cost}")
```

2. **Operation History**
```python
# View operation history
for op in df.semantic.history:
    print(f"{op.op_type}: {op.output_columns}")
```

3. **Validation Rules**
```python
# Add validation rules to any  map or filter operation
validate=["len(output['tags']) <= 5", "output['score'] >= 0"]
```




For more details on configuration options and best practices, refer to:
- [DocETL Best Practices](../best-practices.md)
- [Pipeline Configuration](../concepts/pipelines.md)
- [Output Schemas](../concepts/schemas.md) 