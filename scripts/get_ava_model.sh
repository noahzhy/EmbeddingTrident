#!/usr/bin/env bash

set -euo pipefail

API_BASE="https://aiautomation-api.clobotics.cn/api"
PROJECT_LIST_URL="${API_BASE}/project/get_all"
PROJECT_MODELS_URL="${API_BASE}/projects2models/get_project_models"

AUTH_TOKEN="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3NfdG9rZW4iOiIxMjMiLCJleHAiOjQ4OTYxNDU5OTN9.fjwmVKD5-iyPe1UTKn51nKRjzZPJoZEpyG4G7ypB7i0"
PROJECT_LIST_BODY='{"currentPage":1,"pageSize":500,"searchName":"","userId":"35"}'

for cmd in curl jq; do
	if ! command -v "${cmd}" >/dev/null 2>&1; then
		echo "Error: required command not found: ${cmd}" >&2
		exit 1
	fi
done

post_json() {
	local url="$1"
	local body="$2"

	curl --silent --show-error --fail \
		--request POST "${url}" \
		--header "Content-Type: application/json" \
		--header "authorization: ${AUTH_TOKEN}" \
		--data "${body}"
}

projects_response="$(post_json "${PROJECT_LIST_URL}" "${PROJECT_LIST_BODY}")"
project_api_code="$(echo "${projects_response}" | jq -r '.code // empty')"

if [[ "${project_api_code}" != "0" ]]; then
	echo "Error: get_all API returned code=${project_api_code}" >&2
	echo "${projects_response}" | jq . >&2
	exit 1
fi

project_count="$(echo "${projects_response}" | jq -r '.data.data | length')"
if [[ "${project_count}" == "0" ]]; then
	echo "No projects found."
	exit 0
fi

echo "Total projects: ${project_count}"
echo "========================================"

while IFS=$'\t' read -r project_id project_name; do
	[[ -z "${project_id}" ]] && continue

	echo "Project: ${project_name} (ID: ${project_id})"

	models_body="{\"project_id\":\"${project_id}\"}"
	models_response="$(post_json "${PROJECT_MODELS_URL}" "${models_body}")"
	models_api_code="$(echo "${models_response}" | jq -r '.code // empty')"

	if [[ "${models_api_code}" != "0" ]]; then
		echo "  Error: get_project_models API returned code=${models_api_code}" >&2
		echo "  Raw response: ${models_response}" >&2
		echo
		continue
	fi

	sku_models="$(echo "${models_response}" | jq -r '[.data[]? | select(.model_type_id == 2) | .model_name] | join(", ")')"
	unit_models="$(echo "${models_response}" | jq -r '[.data[]? | select(.model_type_id == 1) | .model_name] | join(", ")')"

	if [[ -z "${sku_models}" ]]; then
		sku_models="(none)"
	fi

	if [[ -z "${unit_models}" ]]; then
		unit_models="(none)"
	fi

	echo "   UNIT: ${unit_models}"
	echo "    SKU: ${sku_models}"
	echo
done < <(echo "${projects_response}" | jq -r '.data.data[]? | [.project_id, .project_name] | @tsv')
