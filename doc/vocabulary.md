# Semantic Layer Vocabulary

Reference definitions for the action naming scheme used in `actions.yml` and task `action_chain` entries.

## Action naming convention

```
action.object_type[.object_facet]
```

## Action verbs

| Verb | Meaning |
|---|---|
| `search` | search/find by semantic or keyword query |
| `get` | retrieve a specific object or list |
| `analyze` | apply filters or computations to return specific data (aggregate, exact value, …) |

## Object types

| Type | Meaning |
|---|---|
| `dataset` | a dataset object on data.gouv.fr |
| `resource` | a resource (file or API endpoint) tied to a dataset |
| `data` | the actual data content of a resource |
| `dataservice` | a third-party API referenced on data.gouv.fr |
| `organization` | a publishing organization |

## Object facets

| Facet | Meaning |
|---|---|
| `resources` | the list of resources belonging to a dataset |
| `info` | basic metadata (title, URL, format, size, description, …) |
| `profile` | tabular schema / column profile (types, stats) — tabular resources only |
| `update_date` | last modification date |
| `create_date` | creation date |

## Notes

- `get.resource.info` and `get.resource.profile` are **distinct actions**:
  - `.info` = basic file metadata via `GET /datasets/{id}/resources/{rid}/`
  - `.profile` = tabular column schema via `GET /resources/{rid}/profile/` (tabular-api)
- `analyze.data` covers both server-side filtering (MCP `query_resource_data`, tabular API) **and** local pandas/code analysis on downloaded data — the evaluator uses the action mapper to distinguish.
- `web_search` / `http_fetch` are **capability tool names**, not semantic actions. They are execution primitives that may implement semantic actions indirectly.
