const JSON_HEADERS = {
    "content-type": "application/json",
};
async function expectJson(response) {
    if (!response.ok) {
        throw new Error(`${response.status} ${response.statusText}`);
    }
    return (await response.json());
}
async function getJson(path) {
    const response = await fetch(path, {
        method: "GET",
        credentials: "same-origin",
        cache: "no-store",
    });
    return expectJson(response);
}
async function postJson(path, body) {
    const response = await fetch(path, {
        method: "POST",
        credentials: "same-origin",
        cache: "no-store",
        headers: JSON_HEADERS,
        body: JSON.stringify(body),
    });
    return expectJson(response);
}
export function parseTriadHandle(raw) {
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
export function encodeTriadHandle(triad) {
    if (!triad) {
        return null;
    }
    return [triad.assetAId, triad.assetBId, triad.assetCId].join(",");
}
export function readExploreRouteState(locationLike = window.location) {
    const params = new URLSearchParams(locationLike.search);
    const mode = params.get("mode") === "learned" ? "learned" : "raw";
    return {
        mode,
        focusId: params.get("focus"),
        triad: parseTriadHandle(params.get("triad")),
    };
}
export function readTriadRoute(locationLike = window.location) {
    return parseTriadHandle(new URLSearchParams(locationLike.search).get("triad"));
}
export function writeExploreRoute(state, historyMode) {
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
    }
    else {
        window.history.replaceState(null, "", href);
    }
}
export function writeTriadRoute(triad, historyMode) {
    const params = new URLSearchParams();
    const encoded = encodeTriadHandle(triad);
    if (encoded) {
        params.set("triad", encoded);
    }
    const href = params.size > 0 ? `/triad?${params.toString()}` : "/triad";
    if (historyMode === "push") {
        window.history.pushState(null, "", href);
    }
    else {
        window.history.replaceState(null, "", href);
    }
}
function exploreQuery(state) {
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
export async function fetchExploreBootstrap(state) {
    return getJson(`/api/explore/bootstrap?${exploreQuery(state)}`);
}
export async function fetchExploreSelection(assetId, state) {
    return getJson(`/api/explore/selection/${encodeURIComponent(assetId)}?${exploreQuery(state)}`);
}
export async function rotateExploreAsset(state, assetId, direction) {
    const body = { assetId, direction };
    return postJson(`/api/explore/rotate?${exploreQuery(state)}`, body);
}
export async function hideExploreAsset(state, assetId) {
    const body = { assetId };
    return postJson(`/api/explore/hide?${exploreQuery(state)}`, body);
}
export async function heartExploreAsset(state, assetId, active) {
    const body = { assetId, active };
    return postJson(`/api/explore/heart?${exploreQuery(state)}`, body);
}
export async function setExploreAssetDomain(state, assetId, label) {
    const body = { assetId, label };
    return postJson(`/api/explore/domain?${exploreQuery(state)}`, body);
}
export async function fetchTriadBootstrap(triad) {
    const params = new URLSearchParams();
    const encoded = encodeTriadHandle(triad);
    if (encoded) {
        params.set("triad", encoded);
    }
    const query = params.toString();
    return getJson(query ? `/api/triad/bootstrap?${query}` : "/api/triad/bootstrap");
}
export async function trainTriad(triad, choice) {
    const body = { triad, choice };
    return postJson("/api/triad/train", body);
}
export async function rotateTriadAsset(triad, assetId, direction) {
    const params = new URLSearchParams();
    const encoded = encodeTriadHandle(triad);
    if (encoded) {
        params.set("triad", encoded);
    }
    const body = { assetId, direction };
    const query = params.toString();
    return postJson(query ? `/api/triad/rotate?${query}` : "/api/triad/rotate", body);
}
export async function hideTriadAsset(triad, assetId) {
    const params = new URLSearchParams();
    const encoded = encodeTriadHandle(triad);
    if (encoded) {
        params.set("triad", encoded);
    }
    const body = { assetId };
    const query = params.toString();
    return postJson(query ? `/api/triad/hide?${query}` : "/api/triad/hide", body);
}
export async function heartTriadAsset(triad, assetId, active) {
    const params = new URLSearchParams();
    const encoded = encodeTriadHandle(triad);
    if (encoded) {
        params.set("triad", encoded);
    }
    const body = { assetId, active };
    const query = params.toString();
    return postJson(query ? `/api/triad/heart?${query}` : "/api/triad/heart", body);
}
export async function setTriadAssetDomain(triad, assetId, label) {
    const params = new URLSearchParams();
    const encoded = encodeTriadHandle(triad);
    if (encoded) {
        params.set("triad", encoded);
    }
    const body = { assetId, label };
    const query = params.toString();
    return postJson(query ? `/api/triad/domain?${query}` : "/api/triad/domain", body);
}
