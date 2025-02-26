const manifest = (() => {
function __memo(fn) {
	let value;
	return () => value ??= (value = fn());
}

return {
	appDir: "_app",
	appPath: "_app",
	assets: new Set([]),
	mimeTypes: {},
	_: {
		client: {"start":"_app/immutable/entry/start.B9ezvl2u.js","app":"_app/immutable/entry/app.Cj4Mo-z4.js","imports":["_app/immutable/entry/start.B9ezvl2u.js","_app/immutable/chunks/client.D7BHBgL3.js","_app/immutable/entry/app.Cj4Mo-z4.js","_app/immutable/chunks/preload-helper.DpQnamwV.js"],"stylesheets":[],"fonts":[],"uses_env_dynamic_public":false},
		nodes: [
			__memo(() => import('./chunks/0-Dn3ctTgk.js')),
			__memo(() => import('./chunks/1-BMFb4diD.js')),
			__memo(() => import('./chunks/2-B4d9dQ-8.js').then(function (n) { return n.aI; }))
		],
		routes: [
			{
				id: "/[...catchall]",
				pattern: /^(?:\/(.*))?\/?$/,
				params: [{"name":"catchall","optional":false,"rest":true,"chained":true}],
				page: { layouts: [0,], errors: [1,], leaf: 2 },
				endpoint: null
			}
		],
		matchers: async () => {
			
			return {  };
		},
		server_assets: {}
	}
}
})();

const prerendered = new Set([]);

const base = "";

export { base, manifest, prerendered };
//# sourceMappingURL=manifest.js.map
