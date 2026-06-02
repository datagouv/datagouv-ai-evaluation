# Semantic Layer Vocabulary

Reference definitions for the action naming scheme used in `actions.yml` and task `action_chain` entries. This file tracks the **vocabulary actually in use**; reserved-but-unused terms are noted as such.

## Action naming convention

```
action.object_type[.object_facet]
```

## Action verbs (in use)

| Verb | Meaning |
|---|---|
| `search` | search/find candidates by semantic or keyword query |
| `get` | retrieve a specific object, a list of objects, or the data behind a resource |

## Object types (in use)

| Type | Meaning |
|---|---|
| `dataset` | a dataset object on data.gouv.fr |
| `dataservice` | a third-party API referenced on data.gouv.fr |
| `resource` | a resource (file or API endpoint) tied to a dataset |
| `data` | the actual data content of a resource |

Reserved (not currently mapped to any action): `organization`.

## Object facets (in use)

| Facet | Meaning |
|---|---|
| `info` | basic metadata (title, URL, format, size, description, …) |
| `resources` | the list of resources belonging to a dataset |
| `profile` | tabular schema / column profile (types, stats) — tabular resources only |
| `openapi_spec` | OpenAPI specification of a dataservice |

## Active action set

The full list of semantic actions currently defined in `agent_eval/semantic_layer/config/actions.yml`:

| Action | Object | Notes |
|---|---|---|
| `search.datasets` | dataset | semantic/keyword search of the catalog |
| `search.dataservices` | dataservice | semantic/keyword search of the catalog |
| `get.dataset.info` | dataset | dataset metadata by id |
| `get.dataset.resources` | dataset | list resources belonging to a dataset |
| `get.resource.info` | resource | file metadata via `GET /datasets/{id}/resources/{rid}/` |
| `get.resource.profile` | resource | tabular column schema via `GET /resources/{rid}/profile/` (tabular-api) |
| `get.data` | resource | fetch actual data — server-side filtering (MCP `query_resource_data`, tabular API) **and** local pandas/code analysis on downloaded data are both classified under this action; the action mapper disambiguates by source |
| `get.dataservice.info` | dataservice | dataservice metadata by id |
| `get.dataservice.openapi_spec` | dataservice | OpenAPI spec of a dataservice |

## Notes

- `get.resource.info` and `get.resource.profile` are **distinct actions**: `.info` returns file metadata, `.profile` returns the tabular column schema.
- `web_search` / `http_fetch` / `execute_python` / `execute_cli` are **capability tool names**, not semantic actions. They are execution primitives that may implement semantic actions indirectly (classified by the LLM-judged action mapper at evaluation time).
