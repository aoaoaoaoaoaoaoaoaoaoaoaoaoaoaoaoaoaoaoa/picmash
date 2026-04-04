import type {
  AssetDomainLabel,
  AssetDomainRequestDto,
  ExploreBootstrapDto,
  ExploreMapMode,
  ExploreSelectionResponseDto,
  HeartAssetRequestDto,
  HideAssetRequestDto,
  RotateAssetRequestDto,
  SimilarityChoice,
  TriadBootstrapDto,
  TriadHandleDto,
  TriadTrainRequestDto,
} from "./contracts";

export type ExploreRouteState = {
  mode: ExploreMapMode;
  focusId: string | null;
  triad: TriadHandleDto | null;
};

const JSON_HEADERS = {
  "content-type": "application/json",
};

async function expectJson<T>(response: Response): Promise<T> {
  if (!response.ok) {
    throw new Error(`${response.status} ${response.statusText}`);
  }
  return (await response.json()) as T;
}

async function getJson<T>(path: string): Promise<T> {
  const response = await fetch(path, {
    method: "GET",
    credentials: "same-origin",
    cache: "no-store",
  });
  return expectJson<T>(response);
}

async function postJson<T>(path: string, body: unknown): Promise<T> {
  const response = await fetch(path, {
    method: "POST",
    credentials: "same-origin",
    cache: "no-store",
    headers: JSON_HEADERS,
    body: JSON.stringify(body),
  });
  return expectJson<T>(response);
}

export function parseTriadHandle(raw: string | null): TriadHandleDto | null {
  if (!raw) {
    return null;
  }
  const parts = raw
    .split(",")
    .map((part) => part.trim())
    .filter(Boolean);
  if (parts.length !== 3) {
    return null;
  }
  const [assetAId, assetBId, assetCId] = parts;
  return { assetAId, assetBId, assetCId };
}

export function encodeTriadHandle(triad: TriadHandleDto | null): string | null {
  if (!triad) {
    return null;
  }
  return [triad.assetAId, triad.assetBId, triad.assetCId].join(",");
}

export function readExploreRouteState(locationLike: Location = window.location): ExploreRouteState {
  const params = new URLSearchParams(locationLike.search);
  const mode = params.get("mode") === "learned" ? "learned" : "raw";
  return {
    mode,
    focusId: params.get("focus"),
    triad: parseTriadHandle(params.get("triad")),
  };
}

export function readTriadRoute(locationLike: Location = window.location): TriadHandleDto | null {
  return parseTriadHandle(new URLSearchParams(locationLike.search).get("triad"));
}

export function writeExploreRoute(
  state: ExploreRouteState,
  historyMode: "push" | "replace",
): void {
  const params = new URLSearchParams();
  params.set("mode", state.mode);
  const triad = encodeTriadHandle(state.triad);
  if (triad) {
    params.set("triad", triad);
  }
  if (state.focusId) {
    params.set("focus", state.focusId);
  }
  const href = `/explore?${params.toString()}`;
  if (historyMode === "push") {
    window.history.pushState(null, "", href);
  } else {
    window.history.replaceState(null, "", href);
  }
}

export function writeTriadRoute(
  triad: TriadHandleDto | null,
  historyMode: "push" | "replace",
): void {
  const params = new URLSearchParams();
  const encoded = encodeTriadHandle(triad);
  if (encoded) {
    params.set("triad", encoded);
  }
  const href = params.size > 0 ? `/triad?${params.toString()}` : "/triad";
  if (historyMode === "push") {
    window.history.pushState(null, "", href);
  } else {
    window.history.replaceState(null, "", href);
  }
}

function exploreQuery(state: ExploreRouteState): string {
  const params = new URLSearchParams();
  params.set("mode", state.mode);
  const triad = encodeTriadHandle(state.triad);
  if (triad) {
    params.set("triad", triad);
  }
  if (state.focusId) {
    params.set("focus", state.focusId);
  }
  return params.toString();
}

export async function fetchExploreBootstrap(
  state: ExploreRouteState,
): Promise<ExploreBootstrapDto> {
  return getJson<ExploreBootstrapDto>(`/api/explore/bootstrap?${exploreQuery(state)}`);
}

export async function fetchExploreSelection(
  assetId: string,
  state: ExploreRouteState,
): Promise<ExploreSelectionResponseDto> {
  return getJson<ExploreSelectionResponseDto>(
    `/api/explore/selection/${encodeURIComponent(assetId)}?${exploreQuery(state)}`,
  );
}

export async function rotateExploreAsset(
  state: ExploreRouteState,
  assetId: string,
  direction: -1 | 1,
): Promise<ExploreBootstrapDto> {
  const body: RotateAssetRequestDto = { assetId, direction };
  return postJson<ExploreBootstrapDto>(
    `/api/explore/rotate?${exploreQuery(state)}`,
    body,
  );
}

export async function hideExploreAsset(
  state: ExploreRouteState,
  assetId: string,
): Promise<ExploreBootstrapDto> {
  const body: HideAssetRequestDto = { assetId };
  return postJson<ExploreBootstrapDto>(`/api/explore/hide?${exploreQuery(state)}`, body);
}

export async function heartExploreAsset(
  state: ExploreRouteState,
  assetId: string,
  active: boolean,
): Promise<ExploreBootstrapDto> {
  const body: HeartAssetRequestDto = { assetId, active };
  return postJson<ExploreBootstrapDto>(`/api/explore/heart?${exploreQuery(state)}`, body);
}

export async function setExploreAssetDomain(
  state: ExploreRouteState,
  assetId: string,
  label: AssetDomainLabel,
): Promise<ExploreBootstrapDto> {
  const body: AssetDomainRequestDto = { assetId, label };
  return postJson<ExploreBootstrapDto>(`/api/explore/domain?${exploreQuery(state)}`, body);
}

export async function fetchTriadBootstrap(
  triad: TriadHandleDto | null,
): Promise<TriadBootstrapDto> {
  const params = new URLSearchParams();
  const encoded = encodeTriadHandle(triad);
  if (encoded) {
    params.set("triad", encoded);
  }
  const query = params.toString();
  return getJson<TriadBootstrapDto>(query ? `/api/triad/bootstrap?${query}` : "/api/triad/bootstrap");
}

export async function trainTriad(
  triad: TriadHandleDto,
  choice: SimilarityChoice,
): Promise<TriadBootstrapDto> {
  const body: TriadTrainRequestDto = { triad, choice };
  return postJson<TriadBootstrapDto>("/api/triad/train", body);
}

export async function rotateTriadAsset(
  triad: TriadHandleDto | null,
  assetId: string,
  direction: -1 | 1,
): Promise<TriadBootstrapDto> {
  const params = new URLSearchParams();
  const encoded = encodeTriadHandle(triad);
  if (encoded) {
    params.set("triad", encoded);
  }
  const body: RotateAssetRequestDto = { assetId, direction };
  const query = params.toString();
  return postJson<TriadBootstrapDto>(query ? `/api/triad/rotate?${query}` : "/api/triad/rotate", body);
}

export async function hideTriadAsset(
  triad: TriadHandleDto | null,
  assetId: string,
): Promise<TriadBootstrapDto> {
  const params = new URLSearchParams();
  const encoded = encodeTriadHandle(triad);
  if (encoded) {
    params.set("triad", encoded);
  }
  const body: HideAssetRequestDto = { assetId };
  const query = params.toString();
  return postJson<TriadBootstrapDto>(query ? `/api/triad/hide?${query}` : "/api/triad/hide", body);
}

export async function heartTriadAsset(
  triad: TriadHandleDto | null,
  assetId: string,
  active: boolean,
): Promise<TriadBootstrapDto> {
  const params = new URLSearchParams();
  const encoded = encodeTriadHandle(triad);
  if (encoded) {
    params.set("triad", encoded);
  }
  const body: HeartAssetRequestDto = { assetId, active };
  const query = params.toString();
  return postJson<TriadBootstrapDto>(
    query ? `/api/triad/heart?${query}` : "/api/triad/heart",
    body,
  );
}

export async function setTriadAssetDomain(
  triad: TriadHandleDto | null,
  assetId: string,
  label: AssetDomainLabel,
): Promise<TriadBootstrapDto> {
  const params = new URLSearchParams();
  const encoded = encodeTriadHandle(triad);
  if (encoded) {
    params.set("triad", encoded);
  }
  const body: AssetDomainRequestDto = { assetId, label };
  const query = params.toString();
  return postJson<TriadBootstrapDto>(
    query ? `/api/triad/domain?${query}` : "/api/triad/domain",
    body,
  );
}
