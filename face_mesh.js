(function () {
  /*

    Copyright The Closure Library Authors.
    SPDX-License-Identifier: Apache-2.0
   */
  "use strict";
  var v;
  function aa(a) {
    var b = 0;
    return function () {
      return b < a.length ? { done: !1, value: a[b++] } : { done: !0 };
    };
  }
  var ba =
    "function" == typeof Object.defineProperties
      ? Object.defineProperty
      : function (a, b, c) {
          if (a == Array.prototype || a == Object.prototype) return a;
          a[b] = c.value;
          return a;
        };
  function ca(a) {
    a = [
      "object" == typeof globalThis && globalThis,
      a,
      "object" == typeof window && window,
      "object" == typeof self && self,
      "object" == typeof global && global,
    ];
    for (var b = 0; b < a.length; ++b) {
      var c = a[b];
      if (c && c.Math == Math) return c;
    }
    throw Error("Cannot find global object");
  }
  var G = ca(this);
  function J(a, b) {
    if (b)
      a: {
        var c = G;
        a = a.split(".");
        for (var d = 0; d < a.length - 1; d++) {
          var e = a[d];
          if (!(e in c)) break a;
          c = c[e];
        }
        a = a[a.length - 1];
        d = c[a];
        b = b(d);
        b != d &&
          null != b &&
          ba(c, a, { configurable: !0, writable: !0, value: b });
      }
  }
  J("Symbol", function (a) {
    function b(g) {
      if (this instanceof b) throw new TypeError("Symbol is not a constructor");
      return new c(d + (g || "") + "_" + e++, g);
    }
    function c(g, f) {
      this.g = g;
      ba(this, "description", { configurable: !0, writable: !0, value: f });
    }
    if (a) return a;
    c.prototype.toString = function () {
      return this.g;
    };
    var d = "jscomp_symbol_" + ((1e9 * Math.random()) >>> 0) + "_",
      e = 0;
    return b;
  });
  J("Symbol.iterator", function (a) {
    if (a) return a;
    a = Symbol("Symbol.iterator");
    for (
      var b =
          "Array Int8Array Uint8Array Uint8ClampedArray Int16Array Uint16Array Int32Array Uint32Array Float32Array Float64Array".split(
            " "
          ),
        c = 0;
      c < b.length;
      c++
    ) {
      var d = G[b[c]];
      "function" === typeof d &&
        "function" != typeof d.prototype[a] &&
        ba(d.prototype, a, {
          configurable: !0,
          writable: !0,
          value: function () {
            return da(aa(this));
          },
        });
    }
    return a;
  });
  function da(a) {
    a = { next: a };
    a[Symbol.iterator] = function () {
      return this;
    };
    return a;
  }
  function K(a) {
    var b =
      "undefined" != typeof Symbol && Symbol.iterator && a[Symbol.iterator];
    return b ? b.call(a) : { next: aa(a) };
  }
  function L(a) {
    if (!(a instanceof Array)) {
      a = K(a);
      for (var b, c = []; !(b = a.next()).done; ) c.push(b.value);
      a = c;
    }
    return a;
  }
  var ea =
      "function" == typeof Object.create
        ? Object.create
        : function (a) {
            function b() {}
            b.prototype = a;
            return new b();
          },
    fa;
  if ("function" == typeof Object.setPrototypeOf) fa = Object.setPrototypeOf;
  else {
    var ha;
    a: {
      var ia = { a: !0 },
        ja = {};
      try {
        ja.__proto__ = ia;
        ha = ja.a;
        break a;
      } catch (a) {}
      ha = !1;
    }
    fa = ha
      ? function (a, b) {
          a.__proto__ = b;
          if (a.__proto__ !== b) throw new TypeError(a + " is not extensible");
          return a;
        }
      : null;
  }
  var ka = fa;
  function M(a, b) {
    a.prototype = ea(b.prototype);
    a.prototype.constructor = a;
    if (ka) ka(a, b);
    else
      for (var c in b)
        if ("prototype" != c)
          if (Object.defineProperties) {
            var d = Object.getOwnPropertyDescriptor(b, c);
            d && Object.defineProperty(a, c, d);
          } else a[c] = b[c];
    a.ea = b.prototype;
  }
  function ma() {
    this.l = !1;
    this.i = null;
    this.h = void 0;
    this.g = 1;
    this.s = this.m = 0;
    this.j = null;
  }
  function na(a) {
    if (a.l) throw new TypeError("Generator is already running");
    a.l = !0;
  }
  ma.prototype.o = function (a) {
    this.h = a;
  };
  function oa(a, b) {
    a.j = { U: b, V: !0 };
    a.g = a.m || a.s;
  }
  ma.prototype.return = function (a) {
    this.j = { return: a };
    this.g = this.s;
  };
  function N(a, b, c) {
    a.g = c;
    return { value: b };
  }
  function pa(a) {
    this.g = new ma();
    this.h = a;
  }
  function qa(a, b) {
    na(a.g);
    var c = a.g.i;
    if (c)
      return ra(
        a,
        "return" in c
          ? c["return"]
          : function (d) {
              return { value: d, done: !0 };
            },
        b,
        a.g.return
      );
    a.g.return(b);
    return sa(a);
  }
  function ra(a, b, c, d) {
    try {
      var e = b.call(a.g.i, c);
      if (!(e instanceof Object))
        throw new TypeError("Iterator result " + e + " is not an object");
      if (!e.done) return (a.g.l = !1), e;
      var g = e.value;
    } catch (f) {
      return (a.g.i = null), oa(a.g, f), sa(a);
    }
    a.g.i = null;
    d.call(a.g, g);
    return sa(a);
  }
  function sa(a) {
    for (; a.g.g; )
      try {
        var b = a.h(a.g);
        if (b) return (a.g.l = !1), { value: b.value, done: !1 };
      } catch (c) {
        (a.g.h = void 0), oa(a.g, c);
      }
    a.g.l = !1;
    if (a.g.j) {
      b = a.g.j;
      a.g.j = null;
      if (b.V) throw b.U;
      return { value: b.return, done: !0 };
    }
    return { value: void 0, done: !0 };
  }
  function ta(a) {
    this.next = function (b) {
      na(a.g);
      a.g.i ? (b = ra(a, a.g.i.next, b, a.g.o)) : (a.g.o(b), (b = sa(a)));
      return b;
    };
    this.throw = function (b) {
      na(a.g);
      a.g.i ? (b = ra(a, a.g.i["throw"], b, a.g.o)) : (oa(a.g, b), (b = sa(a)));
      return b;
    };
    this.return = function (b) {
      return qa(a, b);
    };
    this[Symbol.iterator] = function () {
      return this;
    };
  }
  function O(a, b) {
    b = new ta(new pa(b));
    ka && a.prototype && ka(b, a.prototype);
    return b;
  }
  function ua(a, b) {
    a instanceof String && (a += "");
    var c = 0,
      d = !1,
      e = {
        next: function () {
          if (!d && c < a.length) {
            var g = c++;
            return { value: b(g, a[g]), done: !1 };
          }
          d = !0;
          return { done: !0, value: void 0 };
        },
      };
    e[Symbol.iterator] = function () {
      return e;
    };
    return e;
  }
  var va =
    "function" == typeof Object.assign
      ? Object.assign
      : function (a, b) {
          for (var c = 1; c < arguments.length; c++) {
            var d = arguments[c];
            if (d)
              for (var e in d)
                Object.prototype.hasOwnProperty.call(d, e) && (a[e] = d[e]);
          }
          return a;
        };
  J("Object.assign", function (a) {
    return a || va;
  });
  J("Promise", function (a) {
    function b(f) {
      this.h = 0;
      this.i = void 0;
      this.g = [];
      this.o = !1;
      var h = this.j();
      try {
        f(h.resolve, h.reject);
      } catch (k) {
        h.reject(k);
      }
    }
    function c() {
      this.g = null;
    }
    function d(f) {
      return f instanceof b
        ? f
        : new b(function (h) {
            h(f);
          });
    }
    if (a) return a;
    c.prototype.h = function (f) {
      if (null == this.g) {
        this.g = [];
        var h = this;
        this.i(function () {
          h.l();
        });
      }
      this.g.push(f);
    };
    var e = G.setTimeout;
    c.prototype.i = function (f) {
      e(f, 0);
    };
    c.prototype.l = function () {
      for (; this.g && this.g.length; ) {
        var f = this.g;
        this.g = [];
        for (var h = 0; h < f.length; ++h) {
          var k = f[h];
          f[h] = null;
          try {
            k();
          } catch (l) {
            this.j(l);
          }
        }
      }
      this.g = null;
    };
    c.prototype.j = function (f) {
      this.i(function () {
        throw f;
      });
    };
    b.prototype.j = function () {
      function f(l) {
        return function (n) {
          k || ((k = !0), l.call(h, n));
        };
      }
      var h = this,
        k = !1;
      return { resolve: f(this.C), reject: f(this.l) };
    };
    b.prototype.C = function (f) {
      if (f === this)
        this.l(new TypeError("A Promise cannot resolve to itself"));
      else if (f instanceof b) this.F(f);
      else {
        a: switch (typeof f) {
          case "object":
            var h = null != f;
            break a;
          case "function":
            h = !0;
            break a;
          default:
            h = !1;
        }
        h ? this.u(f) : this.m(f);
      }
    };
    b.prototype.u = function (f) {
      var h = void 0;
      try {
        h = f.then;
      } catch (k) {
        this.l(k);
        return;
      }
      "function" == typeof h ? this.G(h, f) : this.m(f);
    };
    b.prototype.l = function (f) {
      this.s(2, f);
    };
    b.prototype.m = function (f) {
      this.s(1, f);
    };
    b.prototype.s = function (f, h) {
      if (0 != this.h)
        throw Error(
          "Cannot settle(" +
            f +
            ", " +
            h +
            "): Promise already settled in state" +
            this.h
        );
      this.h = f;
      this.i = h;
      2 === this.h && this.D();
      this.A();
    };
    b.prototype.D = function () {
      var f = this;
      e(function () {
        if (f.B()) {
          var h = G.console;
          "undefined" !== typeof h && h.error(f.i);
        }
      }, 1);
    };
    b.prototype.B = function () {
      if (this.o) return !1;
      var f = G.CustomEvent,
        h = G.Event,
        k = G.dispatchEvent;
      if ("undefined" === typeof k) return !0;
      "function" === typeof f
        ? (f = new f("unhandledrejection", { cancelable: !0 }))
        : "function" === typeof h
        ? (f = new h("unhandledrejection", { cancelable: !0 }))
        : ((f = G.document.createEvent("CustomEvent")),
          f.initCustomEvent("unhandledrejection", !1, !0, f));
      f.promise = this;
      f.reason = this.i;
      return k(f);
    };
    b.prototype.A = function () {
      if (null != this.g) {
        for (var f = 0; f < this.g.length; ++f) g.h(this.g[f]);
        this.g = null;
      }
    };
    var g = new c();
    b.prototype.F = function (f) {
      var h = this.j();
      f.J(h.resolve, h.reject);
    };
    b.prototype.G = function (f, h) {
      var k = this.j();
      try {
        f.call(h, k.resolve, k.reject);
      } catch (l) {
        k.reject(l);
      }
    };
    b.prototype.then = function (f, h) {
      function k(w, r) {
        return "function" == typeof w
          ? function (y) {
              try {
                l(w(y));
              } catch (m) {
                n(m);
              }
            }
          : r;
      }
      var l,
        n,
        u = new b(function (w, r) {
          l = w;
          n = r;
        });
      this.J(k(f, l), k(h, n));
      return u;
    };
    b.prototype.catch = function (f) {
      return this.then(void 0, f);
    };
    b.prototype.J = function (f, h) {
      function k() {
        switch (l.h) {
          case 1:
            f(l.i);
            break;
          case 2:
            h(l.i);
            break;
          default:
            throw Error("Unexpected state: " + l.h);
        }
      }
      var l = this;
      null == this.g ? g.h(k) : this.g.push(k);
      this.o = !0;
    };
    b.resolve = d;
    b.reject = function (f) {
      return new b(function (h, k) {
        k(f);
      });
    };
    b.race = function (f) {
      return new b(function (h, k) {
        for (var l = K(f), n = l.next(); !n.done; n = l.next())
          d(n.value).J(h, k);
      });
    };
    b.all = function (f) {
      var h = K(f),
        k = h.next();
      return k.done
        ? d([])
        : new b(function (l, n) {
            function u(y) {
              return function (m) {
                w[y] = m;
                r--;
                0 == r && l(w);
              };
            }
            var w = [],
              r = 0;
            do
              w.push(void 0),
                r++,
                d(k.value).J(u(w.length - 1), n),
                (k = h.next());
            while (!k.done);
          });
    };
    return b;
  });
  J("Object.is", function (a) {
    return a
      ? a
      : function (b, c) {
          return b === c ? 0 !== b || 1 / b === 1 / c : b !== b && c !== c;
        };
  });
  J("Array.prototype.includes", function (a) {
    return a
      ? a
      : function (b, c) {
          var d = this;
          d instanceof String && (d = String(d));
          var e = d.length;
          c = c || 0;
          for (0 > c && (c = Math.max(c + e, 0)); c < e; c++) {
            var g = d[c];
            if (g === b || Object.is(g, b)) return !0;
          }
          return !1;
        };
  });
  J("String.prototype.includes", function (a) {
    return a
      ? a
      : function (b, c) {
          if (null == this)
            throw new TypeError(
              "The 'this' value for String.prototype.includes must not be null or undefined"
            );
          if (b instanceof RegExp)
            throw new TypeError(
              "First argument to String.prototype.includes must not be a regular expression"
            );
          return -1 !== this.indexOf(b, c || 0);
        };
  });
  J("Array.prototype.keys", function (a) {
    return a
      ? a
      : function () {
          return ua(this, function (b) {
            return b;
          });
        };
  });
  var wa = this || self;
  function P(a, b) {
    a = a.split(".");
    var c = wa;
    a[0] in c ||
      "undefined" == typeof c.execScript ||
      c.execScript("var " + a[0]);
    for (var d; a.length && (d = a.shift()); )
      a.length || void 0 === b
        ? c[d] && c[d] !== Object.prototype[d]
          ? (c = c[d])
          : (c = c[d] = {})
        : (c[d] = b);
  }
  function xa(a, b) {
    b = String.fromCharCode.apply(null, b);
    return null == a ? b : a + b;
  }
  var ya,
    za = "undefined" !== typeof TextDecoder,
    Aa,
    Ba = "undefined" !== typeof TextEncoder;
  function Ca(a) {
    if (Ba) a = (Aa || (Aa = new TextEncoder())).encode(a);
    else {
      var b = void 0;
      b = void 0 === b ? !1 : b;
      for (
        var c = 0, d = new Uint8Array(3 * a.length), e = 0;
        e < a.length;
        e++
      ) {
        var g = a.charCodeAt(e);
        if (128 > g) d[c++] = g;
        else {
          if (2048 > g) d[c++] = (g >> 6) | 192;
          else {
            if (55296 <= g && 57343 >= g) {
              if (56319 >= g && e < a.length) {
                var f = a.charCodeAt(++e);
                if (56320 <= f && 57343 >= f) {
                  g = 1024 * (g - 55296) + f - 56320 + 65536;
                  d[c++] = (g >> 18) | 240;
                  d[c++] = ((g >> 12) & 63) | 128;
                  d[c++] = ((g >> 6) & 63) | 128;
                  d[c++] = (g & 63) | 128;
                  continue;
                } else e--;
              }
              if (b) throw Error("Found an unpaired surrogate");
              g = 65533;
            }
            d[c++] = (g >> 12) | 224;
            d[c++] = ((g >> 6) & 63) | 128;
          }
          d[c++] = (g & 63) | 128;
        }
      }
      a = d.subarray(0, c);
    }
    return a;
  }
  var Da = {},
    Ea = null;
  function Fa(a, b) {
    void 0 === b && (b = 0);
    Ga();
    b = Da[b];
    for (
      var c = Array(Math.floor(a.length / 3)), d = b[64] || "", e = 0, g = 0;
      e < a.length - 2;
      e += 3
    ) {
      var f = a[e],
        h = a[e + 1],
        k = a[e + 2],
        l = b[f >> 2];
      f = b[((f & 3) << 4) | (h >> 4)];
      h = b[((h & 15) << 2) | (k >> 6)];
      k = b[k & 63];
      c[g++] = l + f + h + k;
    }
    l = 0;
    k = d;
    switch (a.length - e) {
      case 2:
        (l = a[e + 1]), (k = b[(l & 15) << 2] || d);
      case 1:
        (a = a[e]), (c[g] = b[a >> 2] + b[((a & 3) << 4) | (l >> 4)] + k + d);
    }
    return c.join("");
  }
  function Ha(a) {
    var b = a.length,
      c = (3 * b) / 4;
    c % 3
      ? (c = Math.floor(c))
      : -1 != "=.".indexOf(a[b - 1]) &&
        (c = -1 != "=.".indexOf(a[b - 2]) ? c - 2 : c - 1);
    var d = new Uint8Array(c),
      e = 0;
    Ia(a, function (g) {
      d[e++] = g;
    });
    return d.subarray(0, e);
  }
  function Ia(a, b) {
    function c(k) {
      for (; d < a.length; ) {
        var l = a.charAt(d++),
          n = Ea[l];
        if (null != n) return n;
        if (!/^[\s\xa0]*$/.test(l))
          throw Error("Unknown base64 encoding at char: " + l);
      }
      return k;
    }
    Ga();
    for (var d = 0; ; ) {
      var e = c(-1),
        g = c(0),
        f = c(64),
        h = c(64);
      if (64 === h && -1 === e) break;
      b((e << 2) | (g >> 4));
      64 != f &&
        (b(((g << 4) & 240) | (f >> 2)), 64 != h && b(((f << 6) & 192) | h));
    }
  }
  function Ga() {
    if (!Ea) {
      Ea = {};
      for (
        var a =
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789".split(
              ""
            ),
          b = ["+/=", "+/", "-_=", "-_.", "-_"],
          c = 0;
        5 > c;
        c++
      ) {
        var d = a.concat(b[c].split(""));
        Da[c] = d;
        for (var e = 0; e < d.length; e++) {
          var g = d[e];
          void 0 === Ea[g] && (Ea[g] = e);
        }
      }
    }
  }
  var Ja = "function" === typeof Uint8Array.prototype.slice,
    Ka;
  function La(a, b, c) {
    return b === c
      ? Ka || (Ka = new Uint8Array(0))
      : Ja
      ? a.slice(b, c)
      : new Uint8Array(a.subarray(b, c));
  }
  var Q = 0,
    R = 0;
  function Ma(a, b) {
    b = void 0 === b ? {} : b;
    b = void 0 === b.v ? !1 : b.v;
    this.h = null;
    this.g = this.j = this.l = 0;
    this.m = !1;
    this.v = b;
    a && Na(this, a);
  }
  function Na(a, b) {
    b =
      b.constructor === Uint8Array
        ? b
        : b.constructor === ArrayBuffer
        ? new Uint8Array(b)
        : b.constructor === Array
        ? new Uint8Array(b)
        : b.constructor === String
        ? Ha(b)
        : b instanceof Uint8Array
        ? new Uint8Array(b.buffer, b.byteOffset, b.byteLength)
        : new Uint8Array(0);
    a.h = b;
    a.l = 0;
    a.j = a.h.length;
    a.g = a.l;
  }
  Ma.prototype.reset = function () {
    this.g = this.l;
  };
  function Oa(a) {
    for (var b = 128, c = 0, d = 0, e = 0; 4 > e && 128 <= b; e++)
      (b = a.h[a.g++]), (c |= (b & 127) << (7 * e));
    128 <= b &&
      ((b = a.h[a.g++]), (c |= (b & 127) << 28), (d |= (b & 127) >> 4));
    if (128 <= b)
      for (e = 0; 5 > e && 128 <= b; e++)
        (b = a.h[a.g++]), (d |= (b & 127) << (7 * e + 3));
    if (128 > b) {
      a = c >>> 0;
      b = d >>> 0;
      if ((d = b & 2147483648))
        (a = (~a + 1) >>> 0), (b = ~b >>> 0), 0 == a && (b = (b + 1) >>> 0);
      a = 4294967296 * b + (a >>> 0);
      return d ? -a : a;
    }
    a.m = !0;
  }
  Ma.prototype.i = function () {
    var a = this.h,
      b = a[this.g],
      c = b & 127;
    if (128 > b) return (this.g += 1), c;
    b = a[this.g + 1];
    c |= (b & 127) << 7;
    if (128 > b) return (this.g += 2), c;
    b = a[this.g + 2];
    c |= (b & 127) << 14;
    if (128 > b) return (this.g += 3), c;
    b = a[this.g + 3];
    c |= (b & 127) << 21;
    if (128 > b) return (this.g += 4), c;
    b = a[this.g + 4];
    c |= (b & 15) << 28;
    if (128 > b) return (this.g += 5), c >>> 0;
    this.g += 5;
    128 <= a[this.g++] &&
      128 <= a[this.g++] &&
      128 <= a[this.g++] &&
      128 <= a[this.g++] &&
      this.g++;
    return c;
  };
  Ma.prototype.o = function () {
    var a = this.h[this.g],
      b = this.h[this.g + 1];
    var c = this.h[this.g + 2];
    var d = this.h[this.g + 3];
    this.g += 4;
    c = ((a << 0) | (b << 8) | (c << 16) | (d << 24)) >>> 0;
    a = 2 * (c >> 31) + 1;
    b = (c >>> 23) & 255;
    c &= 8388607;
    return 255 == b
      ? c
        ? NaN
        : Infinity * a
      : 0 == b
      ? a * Math.pow(2, -149) * c
      : a * Math.pow(2, b - 150) * (c + Math.pow(2, 23));
  };
  var Pa = [];
  function Qa() {
    this.g = new Uint8Array(64);
    this.h = 0;
  }
  Qa.prototype.push = function (a) {
    if (!(this.h + 1 < this.g.length)) {
      var b = this.g;
      this.g = new Uint8Array(Math.ceil(1 + 2 * this.g.length));
      this.g.set(b);
    }
    this.g[this.h++] = a;
  };
  Qa.prototype.length = function () {
    return this.h;
  };
  Qa.prototype.end = function () {
    var a = this.g,
      b = this.h;
    this.h = 0;
    return La(a, 0, b);
  };
  function Ra(a, b) {
    for (; 127 < b; ) a.push((b & 127) | 128), (b >>>= 7);
    a.push(b);
  }
  function Sa(a) {
    var b = {},
      c = void 0 === b.N ? !1 : b.N;
    this.o = { v: void 0 === b.v ? !1 : b.v };
    this.N = c;
    b = this.o;
    Pa.length
      ? ((c = Pa.pop()), b && (c.v = b.v), a && Na(c, a), (a = c))
      : (a = new Ma(a, b));
    this.g = a;
    this.m = this.g.g;
    this.h = this.i = this.l = -1;
    this.j = !1;
  }
  Sa.prototype.reset = function () {
    this.g.reset();
    this.h = this.l = -1;
  };
  function S(a) {
    var b = a.g;
    (b = b.g == b.j) ||
      (b = a.j) ||
      ((b = a.g), (b = b.m || 0 > b.g || b.g > b.j));
    if (b) return !1;
    a.m = a.g.g;
    b = a.g.i();
    var c = b & 7;
    if (0 != c && 5 != c && 1 != c && 2 != c && 3 != c && 4 != c)
      return (a.j = !0), !1;
    a.i = b;
    a.l = b >>> 3;
    a.h = c;
    return !0;
  }
  function Ta(a) {
    switch (a.h) {
      case 0:
        if (0 != a.h) Ta(a);
        else {
          for (a = a.g; a.h[a.g] & 128; ) a.g++;
          a.g++;
        }
        break;
      case 1:
        1 != a.h ? Ta(a) : ((a = a.g), (a.g += 8));
        break;
      case 2:
        if (2 != a.h) Ta(a);
        else {
          var b = a.g.i();
          a = a.g;
          a.g += b;
        }
        break;
      case 5:
        5 != a.h ? Ta(a) : ((a = a.g), (a.g += 4));
        break;
      case 3:
        b = a.l;
        do {
          if (!S(a)) {
            a.j = !0;
            break;
          }
          if (4 == a.h) {
            a.l != b && (a.j = !0);
            break;
          }
          Ta(a);
        } while (1);
        break;
      default:
        a.j = !0;
    }
  }
  function Ua(a, b, c) {
    var d = a.g.j,
      e = a.g.i(),
      g = a.g.g + e;
    a.g.j = g;
    c(b, a);
    c = g - a.g.g;
    if (0 !== c)
      throw Error(
        "Message parsing ended unexpectedly. Expected to read " +
          e +
          " bytes, instead read " +
          (e - c) +
          " bytes, either the data ended unexpectedly or the message misreported its own length"
      );
    a.g.g = g;
    a.g.j = d;
    return b;
  }
  function T(a) {
    return a.g.o();
  }
  function Va(a) {
    var b = a.g.i();
    a = a.g;
    var c = a.g;
    a.g += b;
    a = a.h;
    var d;
    if (za)
      (d = ya) || (d = ya = new TextDecoder("utf-8", { fatal: !1 })),
        (d = d.decode(a.subarray(c, c + b)));
    else {
      b = c + b;
      for (var e = [], g = null, f, h, k; c < b; )
        (f = a[c++]),
          128 > f
            ? e.push(f)
            : 224 > f
            ? c >= b
              ? e.push(65533)
              : ((h = a[c++]),
                194 > f || 128 !== (h & 192)
                  ? (c--, e.push(65533))
                  : e.push(((f & 31) << 6) | (h & 63)))
            : 240 > f
            ? c >= b - 1
              ? e.push(65533)
              : ((h = a[c++]),
                128 !== (h & 192) ||
                (224 === f && 160 > h) ||
                (237 === f && 160 <= h) ||
                128 !== ((d = a[c++]) & 192)
                  ? (c--, e.push(65533))
                  : e.push(((f & 15) << 12) | ((h & 63) << 6) | (d & 63)))
            : 244 >= f
            ? c >= b - 2
              ? e.push(65533)
              : ((h = a[c++]),
                128 !== (h & 192) ||
                0 !== ((f << 28) + (h - 144)) >> 30 ||
                128 !== ((d = a[c++]) & 192) ||
                128 !== ((k = a[c++]) & 192)
                  ? (c--, e.push(65533))
                  : ((f =
                      ((f & 7) << 18) |
                      ((h & 63) << 12) |
                      ((d & 63) << 6) |
                      (k & 63)),
                    (f -= 65536),
                    e.push(((f >> 10) & 1023) + 55296, (f & 1023) + 56320)))
            : e.push(65533),
          8192 <= e.length && ((g = xa(g, e)), (e.length = 0));
      d = xa(g, e);
    }
    return d;
  }
  function Wa(a, b, c) {
    var d = a.g.i();
    for (d = a.g.g + d; a.g.g < d; ) c.push(b.call(a.g));
  }
  function Xa(a, b) {
    2 == a.h ? Wa(a, Ma.prototype.o, b) : b.push(T(a));
  }
  function Ya() {
    this.h = [];
    this.i = 0;
    this.g = new Qa();
  }
  function Za(a, b) {
    0 !== b.length && (a.h.push(b), (a.i += b.length));
  }
  function $a(a) {
    var b = a.i + a.g.length();
    if (0 === b) return new Uint8Array(0);
    b = new Uint8Array(b);
    for (var c = a.h, d = c.length, e = 0, g = 0; g < d; g++) {
      var f = c[g];
      0 !== f.length && (b.set(f, e), (e += f.length));
    }
    c = a.g;
    d = c.h;
    0 !== d && (b.set(c.g.subarray(0, d), e), (c.h = 0));
    a.h = [b];
    return b;
  }
  function U(a, b, c) {
    if (null != c) {
      Ra(a.g, 8 * b + 5);
      a = a.g;
      var d = c;
      d = (c = 0 > d ? 1 : 0) ? -d : d;
      0 === d
        ? 0 < 1 / d
          ? (Q = R = 0)
          : ((R = 0), (Q = 2147483648))
        : isNaN(d)
        ? ((R = 0), (Q = 2147483647))
        : 3.4028234663852886e38 < d
        ? ((R = 0), (Q = ((c << 31) | 2139095040) >>> 0))
        : 1.1754943508222875e-38 > d
        ? ((d = Math.round(d / Math.pow(2, -149))),
          (R = 0),
          (Q = ((c << 31) | d) >>> 0))
        : ((b = Math.floor(Math.log(d) / Math.LN2)),
          (d *= Math.pow(2, -b)),
          (d = Math.round(8388608 * d)),
          16777216 <= d && ++b,
          (R = 0),
          (Q = ((c << 31) | ((b + 127) << 23) | (d & 8388607)) >>> 0));
      c = Q;
      a.push((c >>> 0) & 255);
      a.push((c >>> 8) & 255);
      a.push((c >>> 16) & 255);
      a.push((c >>> 24) & 255);
    }
  }
  var ab = "function" === typeof Uint8Array;
  function bb(a, b, c) {
    if (null != a)
      return "object" === typeof a
        ? ab && a instanceof Uint8Array
          ? c(a)
          : cb(a, b, c)
        : b(a);
  }
  function cb(a, b, c) {
    if (Array.isArray(a)) {
      for (var d = Array(a.length), e = 0; e < a.length; e++)
        d[e] = bb(a[e], b, c);
      Array.isArray(a) && a.W && db(d);
      return d;
    }
    d = {};
    for (e in a) d[e] = bb(a[e], b, c);
    return d;
  }
  function eb(a) {
    return "number" === typeof a ? (isFinite(a) ? a : String(a)) : a;
  }
  var fb = { W: { value: !0, configurable: !0 } };
  function db(a) {
    Array.isArray(a) && !Object.isFrozen(a) && Object.defineProperties(a, fb);
    return a;
  }
  var gb;
  function V(a, b, c) {
    var d = gb;
    gb = null;
    a || (a = d);
    d = this.constructor.ca;
    a || (a = d ? [d] : []);
    this.j = d ? 0 : -1;
    this.m = this.g = null;
    this.h = a;
    a: {
      d = this.h.length;
      a = d - 1;
      if (
        d &&
        ((d = this.h[a]),
        !(
          null === d ||
          "object" != typeof d ||
          Array.isArray(d) ||
          (ab && d instanceof Uint8Array)
        ))
      ) {
        this.l = a - this.j;
        this.i = d;
        break a;
      }
      void 0 !== b && -1 < b
        ? ((this.l = Math.max(b, a + 1 - this.j)), (this.i = null))
        : (this.l = Number.MAX_VALUE);
    }
    if (c)
      for (b = 0; b < c.length; b++)
        (a = c[b]),
          a < this.l
            ? ((a += this.j), (d = this.h[a]) ? db(d) : (this.h[a] = hb))
            : (ib(this), (d = this.i[a]) ? db(d) : (this.i[a] = hb));
  }
  var hb = Object.freeze(db([]));
  function ib(a) {
    var b = a.l + a.j;
    a.h[b] || (a.i = a.h[b] = {});
  }
  function W(a, b, c) {
    return -1 === b
      ? null
      : (void 0 === c ? 0 : c) || b >= a.l
      ? a.i
        ? a.i[b]
        : void 0
      : a.h[b + a.j];
  }
  function jb(a, b) {
    var c = void 0 === c ? !1 : c;
    var d = W(a, b, c);
    null == d && (d = hb);
    d === hb && ((d = db([])), X(a, b, d, c));
    return d;
  }
  function kb(a) {
    var b = jb(a, 3);
    a.m || (a.m = {});
    if (!a.m[3]) {
      for (var c = 0; c < b.length; c++) b[c] = +b[c];
      a.m[3] = !0;
    }
    return b;
  }
  function lb(a, b, c) {
    a = W(a, b);
    return null == a ? c : a;
  }
  function Y(a, b, c) {
    a = W(a, b);
    a = null == a ? a : +a;
    return null == a ? (void 0 === c ? 0 : c) : a;
  }
  function X(a, b, c, d) {
    (void 0 === d ? 0 : d) || b >= a.l
      ? (ib(a), (a.i[b] = c))
      : (a.h[b + a.j] = c);
  }
  function mb(a, b, c) {
    if (-1 === c) return null;
    a.g || (a.g = {});
    if (!a.g[c]) {
      var d = W(a, c, !1);
      d && (a.g[c] = new b(d));
    }
    return a.g[c];
  }
  function nb(a, b) {
    a.g || (a.g = {});
    var c = a.g[1];
    if (!c) {
      var d = jb(a, 1);
      c = [];
      for (var e = 0; e < d.length; e++) c[e] = new b(d[e]);
      a.g[1] = c;
    }
    return c;
  }
  function ob(a, b, c) {
    var d = void 0 === d ? !1 : d;
    a.g || (a.g = {});
    var e = c ? pb(c, !1) : c;
    a.g[b] = c;
    X(a, b, e, d);
  }
  function qb(a, b, c, d) {
    var e = nb(a, c);
    b = b ? b : new c();
    a = jb(a, 1);
    void 0 != d
      ? (e.splice(d, 0, b), a.splice(d, 0, pb(b, !1)))
      : (e.push(b), a.push(pb(b, !1)));
  }
  V.prototype.toJSON = function () {
    var a = pb(this, !1);
    return cb(a, eb, Fa);
  };
  function pb(a, b) {
    if (a.g)
      for (var c in a.g) {
        var d = a.g[c];
        if (Array.isArray(d))
          for (var e = 0; e < d.length; e++) d[e] && pb(d[e], b);
        else d && pb(d, b);
      }
    return a.h;
  }
  V.prototype.toString = function () {
    return pb(this, !1).toString();
  };
  function rb(a, b) {
    if ((a = a.o)) {
      Za(b, b.g.end());
      for (var c = 0; c < a.length; c++) Za(b, a[c]);
    }
  }
  function sb(a, b) {
    if (4 == b.h) return !1;
    var c = b.m;
    Ta(b);
    b.N || ((b = La(b.g.h, c, b.g.g)), (c = a.o) ? c.push(b) : (a.o = [b]));
    return !0;
  }
  function tb(a) {
    V.call(this, a, -1, ub);
  }
  M(tb, V);
  tb.prototype.getRows = function () {
    return W(this, 1);
  };
  tb.prototype.getCols = function () {
    return W(this, 2);
  };
  tb.prototype.getPackedDataList = function () {
    return kb(this);
  };
  tb.prototype.getLayout = function () {
    return lb(this, 4, 0);
  };
  function vb(a, b) {
    for (; S(b); )
      switch (b.i) {
        case 8:
          var c = b.g.i();
          X(a, 1, c);
          break;
        case 16:
          c = b.g.i();
          X(a, 2, c);
          break;
        case 29:
        case 26:
          Xa(b, a.getPackedDataList());
          break;
        case 32:
          c = Oa(b.g);
          X(a, 4, c);
          break;
        default:
          if (!sb(a, b)) return a;
      }
    return a;
  }
  var ub = [3];
  function Z(a, b) {
    var c = void 0;
    return new (c || (c = Promise))(function (d, e) {
      function g(k) {
        try {
          h(b.next(k));
        } catch (l) {
          e(l);
        }
      }
      function f(k) {
        try {
          h(b["throw"](k));
        } catch (l) {
          e(l);
        }
      }
      function h(k) {
        k.done
          ? d(k.value)
          : new c(function (l) {
              l(k.value);
            }).then(g, f);
      }
      h((b = b.apply(a, void 0)).next());
    });
  }
  function wb(a) {
    V.call(this, a);
  }
  M(wb, V);
  function xb(a, b) {
    for (; S(b); )
      switch (b.i) {
        case 8:
          var c = b.g.i();
          X(a, 1, c);
          break;
        case 21:
          c = T(b);
          X(a, 2, c);
          break;
        case 26:
          c = Va(b);
          X(a, 3, c);
          break;
        case 34:
          c = Va(b);
          X(a, 4, c);
          break;
        default:
          if (!sb(a, b)) return a;
      }
    return a;
  }
  function yb(a) {
    V.call(this, a, -1, zb);
  }
  M(yb, V);
  yb.prototype.addClassification = function (a, b) {
    qb(this, a, wb, b);
    return this;
  };
  var zb = [1];
  function Ab(a) {
    V.call(this, a);
  }
  M(Ab, V);
  function Bb(a, b) {
    for (; S(b); )
      switch (b.i) {
        case 13:
          var c = T(b);
          X(a, 1, c);
          break;
        case 21:
          c = T(b);
          X(a, 2, c);
          break;
        case 29:
          c = T(b);
          X(a, 3, c);
          break;
        case 37:
          c = T(b);
          X(a, 4, c);
          break;
        case 45:
          c = T(b);
          X(a, 5, c);
          break;
        default:
          if (!sb(a, b)) return a;
      }
    return a;
  }
  function Cb(a) {
    V.call(this, a, -1, Db);
  }
  M(Cb, V);
  function Eb(a) {
    a: {
      var b = new Cb();
      for (a = new Sa(a); S(a); )
        switch (a.i) {
          case 10:
            var c = Ua(a, new Ab(), Bb);
            qb(b, c, Ab, void 0);
            break;
          default:
            if (!sb(b, a)) break a;
        }
    }
    return b;
  }
  var Db = [1];
  function Fb(a) {
    V.call(this, a);
  }
  M(Fb, V);
  function Gb(a) {
    V.call(this, a, -1, Hb);
  }
  M(Gb, V);
  Gb.prototype.getVertexType = function () {
    return lb(this, 1, 0);
  };
  Gb.prototype.getPrimitiveType = function () {
    return lb(this, 2, 0);
  };
  Gb.prototype.getVertexBufferList = function () {
    return kb(this);
  };
  Gb.prototype.getIndexBufferList = function () {
    return jb(this, 4);
  };
  function Ib(a, b) {
    for (; S(b); )
      switch (b.i) {
        case 8:
          var c = Oa(b.g);
          X(a, 1, c);
          break;
        case 16:
          c = Oa(b.g);
          X(a, 2, c);
          break;
        case 29:
        case 26:
          Xa(b, a.getVertexBufferList());
          break;
        case 32:
        case 34:
          c = b;
          var d = a.getIndexBufferList();
          2 == c.h ? Wa(c, Ma.prototype.i, d) : d.push(c.g.i());
          break;
        default:
          if (!sb(a, b)) return a;
      }
    return a;
  }
  var Hb = [3, 4];
  function Jb(a) {
    V.call(this, a);
  }
  M(Jb, V);
  Jb.prototype.getMesh = function () {
    return mb(this, Gb, 1);
  };
  Jb.prototype.getPoseTransformMatrix = function () {
    return mb(this, tb, 2);
  };
  function Kb(a) {
    a: {
      var b = new Jb();
      for (a = new Sa(a); S(a); )
        switch (a.i) {
          case 10:
            var c = Ua(a, new Gb(), Ib);
            ob(b, 1, c);
            break;
          case 18:
            c = Ua(a, new tb(), vb);
            ob(b, 2, c);
            break;
          default:
            if (!sb(b, a)) break a;
        }
    }
    return b;
  }
  function Lb(a, b, c) {
    c = a.createShader(0 === c ? a.VERTEX_SHADER : a.FRAGMENT_SHADER);
    a.shaderSource(c, b);
    a.compileShader(c);
    if (!a.getShaderParameter(c, a.COMPILE_STATUS))
      throw Error(
        "Could not compile WebGL shader.\n\n" + a.getShaderInfoLog(c)
      );
    return c;
  }
  function Mb(a) {
    return nb(a, wb).map(function (b) {
      return {
        index: lb(b, 1, 0),
        Y: Y(b, 2),
        label: null != W(b, 3) ? lb(b, 3, "") : void 0,
        displayName: null != W(b, 4) ? lb(b, 4, "") : void 0,
      };
    });
  }
  function Nb(a) {
    return {
      x: Y(a, 1),
      y: Y(a, 2),
      z: Y(a, 3),
      visibility: null != W(a, 4) ? Y(a, 4) : void 0,
    };
  }
  function Ob(a, b) {
    this.h = a;
    this.g = b;
    this.l = 0;
  }
  function Pb(a, b, c) {
    Qb(a, b);
    if ("function" === typeof a.g.canvas.transferToImageBitmap)
      return Promise.resolve(a.g.canvas.transferToImageBitmap());
    if (c) return Promise.resolve(a.g.canvas);
    if ("function" === typeof createImageBitmap)
      return createImageBitmap(a.g.canvas);
    void 0 === a.i && (a.i = document.createElement("canvas"));
    return new Promise(function (d) {
      a.i.height = a.g.canvas.height;
      a.i.width = a.g.canvas.width;
      a.i
        .getContext("2d", {})
        .drawImage(a.g.canvas, 0, 0, a.g.canvas.width, a.g.canvas.height);
      d(a.i);
    });
  }
  function Qb(a, b) {
    var c = a.g;
    if (void 0 === a.m) {
      var d = Lb(
          c,
          "\n  attribute vec2 aVertex;\n  attribute vec2 aTex;\n  varying vec2 vTex;\n  void main(void) {\n    gl_Position = vec4(aVertex, 0.0, 1.0);\n    vTex = aTex;\n  }",
          0
        ),
        e = Lb(
          c,
          "\n  precision mediump float;\n  varying vec2 vTex;\n  uniform sampler2D sampler0;\n  void main(){\n    gl_FragColor = texture2D(sampler0, vTex);\n  }",
          1
        ),
        g = c.createProgram();
      c.attachShader(g, d);
      c.attachShader(g, e);
      c.linkProgram(g);
      if (!c.getProgramParameter(g, c.LINK_STATUS))
        throw Error(
          "Could not compile WebGL program.\n\n" + c.getProgramInfoLog(g)
        );
      d = a.m = g;
      c.useProgram(d);
      e = c.getUniformLocation(d, "sampler0");
      a.j = {
        I: c.getAttribLocation(d, "aVertex"),
        H: c.getAttribLocation(d, "aTex"),
        da: e,
      };
      a.s = c.createBuffer();
      c.bindBuffer(c.ARRAY_BUFFER, a.s);
      c.enableVertexAttribArray(a.j.I);
      c.vertexAttribPointer(a.j.I, 2, c.FLOAT, !1, 0, 0);
      c.bufferData(
        c.ARRAY_BUFFER,
        new Float32Array([-1, -1, -1, 1, 1, 1, 1, -1]),
        c.STATIC_DRAW
      );
      c.bindBuffer(c.ARRAY_BUFFER, null);
      a.o = c.createBuffer();
      c.bindBuffer(c.ARRAY_BUFFER, a.o);
      c.enableVertexAttribArray(a.j.H);
      c.vertexAttribPointer(a.j.H, 2, c.FLOAT, !1, 0, 0);
      c.bufferData(
        c.ARRAY_BUFFER,
        new Float32Array([0, 1, 0, 0, 1, 0, 1, 1]),
        c.STATIC_DRAW
      );
      c.bindBuffer(c.ARRAY_BUFFER, null);
      c.uniform1i(e, 0);
    }
    d = a.j;
    c.useProgram(a.m);
    c.canvas.width = b.width;
    c.canvas.height = b.height;
    c.viewport(0, 0, b.width, b.height);
    c.activeTexture(c.TEXTURE0);
    a.h.bindTexture2d(b.glName);
    c.enableVertexAttribArray(d.I);
    c.bindBuffer(c.ARRAY_BUFFER, a.s);
    c.vertexAttribPointer(d.I, 2, c.FLOAT, !1, 0, 0);
    c.enableVertexAttribArray(d.H);
    c.bindBuffer(c.ARRAY_BUFFER, a.o);
    c.vertexAttribPointer(d.H, 2, c.FLOAT, !1, 0, 0);
    c.bindFramebuffer(
      c.DRAW_FRAMEBUFFER ? c.DRAW_FRAMEBUFFER : c.FRAMEBUFFER,
      null
    );
    c.clearColor(0, 0, 0, 0);
    c.clear(c.COLOR_BUFFER_BIT);
    c.colorMask(!0, !0, !0, !0);
    c.drawArrays(c.TRIANGLE_FAN, 0, 4);
    c.disableVertexAttribArray(d.I);
    c.disableVertexAttribArray(d.H);
    c.bindBuffer(c.ARRAY_BUFFER, null);
    a.h.bindTexture2d(0);
  }
  function Rb(a) {
    this.g = a;
  }
  var Sb = new Uint8Array([
    0, 97, 115, 109, 1, 0, 0, 0, 1, 4, 1, 96, 0, 0, 3, 2, 1, 0, 10, 9, 1, 7, 0,
    65, 0, 253, 15, 26, 11,
  ]);
  function Tb(a, b) {
    return b + a;
  }
  function Ub(a, b) {
    window[a] = b;
  }
  function Vb(a) {
    var b = document.createElement("script");
    b.setAttribute("src", a);
    b.setAttribute("crossorigin", "anonymous");
    return new Promise(function (c) {
      b.addEventListener(
        "load",
        function () {
          c();
        },
        !1
      );
      b.addEventListener(
        "error",
        function () {
          c();
        },
        !1
      );
      document.body.appendChild(b);
    });
  }
  function Wb() {
    return Z(this, function b() {
      return O(b, function (c) {
        switch (c.g) {
          case 1:
            return (c.m = 2), N(c, WebAssembly.instantiate(Sb), 4);
          case 4:
            c.g = 3;
            c.m = 0;
            break;
          case 2:
            return (c.m = 0), (c.j = null), c.return(!1);
          case 3:
            return c.return(!0);
        }
      });
    });
  }
  function Xb(a) {
    this.g = a;
    this.listeners = {};
    this.j = {};
    this.F = {};
    this.m = {};
    this.s = {};
    this.G = this.o = this.R = !0;
    this.C = Promise.resolve();
    this.P = "";
    this.B = {};
    this.locateFile = (a && a.locateFile) || Tb;
    if ("object" === typeof window)
      var b =
        window.location.pathname
          .toString()
          .substring(0, window.location.pathname.toString().lastIndexOf("/")) +
        "/";
    else if ("undefined" !== typeof location)
      b =
        location.pathname
          .toString()
          .substring(0, location.pathname.toString().lastIndexOf("/")) + "/";
    else
      throw Error(
        "solutions can only be loaded on a web page or in a web worker"
      );
    this.S = b;
    if (a.options) {
      b = K(Object.keys(a.options));
      for (var c = b.next(); !c.done; c = b.next()) {
        c = c.value;
        var d = a.options[c].default;
        void 0 !== d && (this.j[c] = "function" === typeof d ? d() : d);
      }
    }
  }
  v = Xb.prototype;
  v.close = function () {
    this.i && this.i.delete();
    return Promise.resolve();
  };
  function Yb(a, b) {
    return void 0 === a.g.files
      ? []
      : "function" === typeof a.g.files
      ? a.g.files(b)
      : a.g.files;
  }
  function Zb(a) {
    return Z(a, function c() {
      var d = this,
        e,
        g,
        f,
        h,
        k,
        l,
        n,
        u,
        w,
        r,
        y;
      return O(c, function (m) {
        switch (m.g) {
          case 1:
            e = d;
            if (!d.R) return m.return();
            g = Yb(d, d.j);
            return N(m, Wb(), 2);
          case 2:
            f = m.h;
            if ("object" === typeof window)
              return (
                Ub("createMediapipeSolutionsWasm", {
                  locateFile: d.locateFile,
                }),
                Ub("createMediapipeSolutionsPackedAssets", {
                  locateFile: d.locateFile,
                }),
                (l = g.filter(function (t) {
                  return void 0 !== t.data;
                })),
                (n = g.filter(function (t) {
                  return void 0 === t.data;
                })),
                (u = Promise.all(
                  l.map(function (t) {
                    var x = $b(e, t.url);
                    if (void 0 !== t.path) {
                      var z = t.path;
                      x = x.then(function (E) {
                        e.overrideFile(z, E);
                        return Promise.resolve(E);
                      });
                    }
                    return x;
                  })
                )),
                (w = Promise.all(
                  n.map(function (t) {
                    return void 0 === t.simd || (t.simd && f) || (!t.simd && !f)
                      ? Vb(e.locateFile(t.url, e.S))
                      : Promise.resolve();
                  })
                ).then(function () {
                  return Z(e, function x() {
                    var z,
                      E,
                      F = this;
                    return O(x, function (I) {
                      if (1 == I.g)
                        return (
                          (z = window.createMediapipeSolutionsWasm),
                          (E = window.createMediapipeSolutionsPackedAssets),
                          N(I, z(E), 2)
                        );
                      F.h = I.h;
                      I.g = 0;
                    });
                  });
                })),
                (r = (function () {
                  return Z(e, function x() {
                    var z = this;
                    return O(x, function (E) {
                      z.g.graph && z.g.graph.url
                        ? (E = N(E, $b(z, z.g.graph.url), 0))
                        : ((E.g = 0), (E = void 0));
                      return E;
                    });
                  });
                })()),
                N(m, Promise.all([w, u, r]), 7)
              );
            if ("function" !== typeof importScripts)
              throw Error(
                "solutions can only be loaded on a web page or in a web worker"
              );
            h = g
              .filter(function (t) {
                return void 0 === t.simd || (t.simd && f) || (!t.simd && !f);
              })
              .map(function (t) {
                return e.locateFile(t.url, e.S);
              });
            importScripts.apply(null, L(h));
            return N(m, createMediapipeSolutionsWasm(Module), 6);
          case 6:
            d.h = m.h;
            d.l = new OffscreenCanvas(1, 1);
            d.h.canvas = d.l;
            k = d.h.GL.createContext(d.l, {
              antialias: !1,
              alpha: !1,
              ba: "undefined" !== typeof WebGL2RenderingContext ? 2 : 1,
            });
            d.h.GL.makeContextCurrent(k);
            m.g = 4;
            break;
          case 7:
            d.l = document.createElement("canvas");
            y = d.l.getContext("webgl2", {});
            if (!y && ((y = d.l.getContext("webgl", {})), !y))
              return (
                alert(
                  "Failed to create WebGL canvas context when passing video frame."
                ),
                m.return()
              );
            d.D = y;
            d.h.canvas = d.l;
            d.h.createContext(d.l, !0, !0, {});
          case 4:
            (d.i = new d.h.SolutionWasm()), (d.R = !1), (m.g = 0);
        }
      });
    });
  }
  function ac(a) {
    return Z(a, function c() {
      var d = this,
        e,
        g,
        f,
        h,
        k,
        l,
        n,
        u;
      return O(c, function (w) {
        if (1 == w.g) {
          if (d.g.graph && d.g.graph.url && d.P === d.g.graph.url)
            return w.return();
          d.o = !0;
          if (!d.g.graph || !d.g.graph.url) {
            w.g = 2;
            return;
          }
          d.P = d.g.graph.url;
          return N(w, $b(d, d.g.graph.url), 3);
        }
        2 != w.g && ((e = w.h), d.i.loadGraph(e));
        g = K(Object.keys(d.B));
        for (f = g.next(); !f.done; f = g.next())
          (h = f.value), d.i.overrideFile(h, d.B[h]);
        d.B = {};
        if (d.g.listeners)
          for (k = K(d.g.listeners), l = k.next(); !l.done; l = k.next())
            (n = l.value), bc(d, n);
        u = d.j;
        d.j = {};
        d.setOptions(u);
        w.g = 0;
      });
    });
  }
  v.reset = function () {
    return Z(this, function b() {
      var c = this;
      return O(b, function (d) {
        c.i && (c.i.reset(), (c.m = {}), (c.s = {}));
        d.g = 0;
      });
    });
  };
  v.setOptions = function (a, b) {
    var c = this;
    if ((b = b || this.g.options)) {
      for (
        var d = [], e = [], g = {}, f = K(Object.keys(a)), h = f.next();
        !h.done;
        g = { K: g.K, L: g.L }, h = f.next()
      ) {
        var k = h.value;
        (k in this.j && this.j[k] === a[k]) ||
          ((this.j[k] = a[k]),
          (h = b[k]),
          void 0 !== h &&
            (h.onChange &&
              ((g.K = h.onChange),
              (g.L = a[k]),
              d.push(
                (function (l) {
                  return function () {
                    return Z(c, function u() {
                      var w,
                        r = this;
                      return O(u, function (y) {
                        if (1 == y.g) return N(y, l.K(l.L), 2);
                        w = y.h;
                        !0 === w && (r.o = !0);
                        y.g = 0;
                      });
                    });
                  };
                })(g)
              )),
            h.graphOptionXref &&
              ((k = {
                valueNumber: 1 === h.type ? a[k] : 0,
                valueBoolean: 0 === h.type ? a[k] : !1,
                valueString: 2 === h.type ? a[k] : "",
              }),
              (h = Object.assign(
                Object.assign(
                  Object.assign({}, { calculatorName: "", calculatorIndex: 0 }),
                  h.graphOptionXref
                ),
                k
              )),
              e.push(h))));
      }
      if (0 !== d.length || 0 !== e.length)
        (this.o = !0),
          (this.A = (void 0 === this.A ? [] : this.A).concat(e)),
          (this.u = (void 0 === this.u ? [] : this.u).concat(d));
    }
  };
  function cc(a) {
    return Z(a, function c() {
      var d = this,
        e,
        g,
        f,
        h,
        k,
        l,
        n;
      return O(c, function (u) {
        switch (u.g) {
          case 1:
            if (!d.o) return u.return();
            if (!d.u) {
              u.g = 2;
              break;
            }
            e = K(d.u);
            g = e.next();
          case 3:
            if (g.done) {
              u.g = 5;
              break;
            }
            f = g.value;
            return N(u, f(), 4);
          case 4:
            g = e.next();
            u.g = 3;
            break;
          case 5:
            d.u = void 0;
          case 2:
            if (d.A) {
              h = new d.h.GraphOptionChangeRequestList();
              k = K(d.A);
              for (l = k.next(); !l.done; l = k.next())
                (n = l.value), h.push_back(n);
              d.i.changeOptions(h);
              h.delete();
              d.A = void 0;
            }
            d.o = !1;
            u.g = 0;
        }
      });
    });
  }
  v.initialize = function () {
    return Z(this, function b() {
      var c = this;
      return O(b, function (d) {
        return 1 == d.g
          ? N(d, Zb(c), 2)
          : 3 != d.g
          ? N(d, ac(c), 3)
          : N(d, cc(c), 0);
      });
    });
  };
  function $b(a, b) {
    return Z(a, function d() {
      var e = this,
        g,
        f;
      return O(d, function (h) {
        if (b in e.F) return h.return(e.F[b]);
        g = e.locateFile(b, "");
        f = fetch(g).then(function (k) {
          return k.arrayBuffer();
        });
        e.F[b] = f;
        return h.return(f);
      });
    });
  }
  v.overrideFile = function (a, b) {
    this.i ? this.i.overrideFile(a, b) : (this.B[a] = b);
  };
  v.clearOverriddenFiles = function () {
    this.B = {};
    this.i && this.i.clearOverriddenFiles();
  };
  v.send = function (a, b) {
    return Z(this, function d() {
      var e = this,
        g,
        f,
        h,
        k,
        l,
        n,
        u,
        w,
        r;
      return O(d, function (y) {
        switch (y.g) {
          case 1:
            if (!e.g.inputs) return y.return();
            g = 1e3 * (void 0 === b || null === b ? performance.now() : b);
            return N(y, e.C, 2);
          case 2:
            return N(y, e.initialize(), 3);
          case 3:
            f = new e.h.PacketDataList();
            h = K(Object.keys(a));
            for (k = h.next(); !k.done; k = h.next())
              if (((l = k.value), (n = e.g.inputs[l]))) {
                a: {
                  var m = e;
                  var t = a[l];
                  switch (n.type) {
                    case "video":
                      var x = m.m[n.stream];
                      x || ((x = new Ob(m.h, m.D)), (m.m[n.stream] = x));
                      m = x;
                      0 === m.l && (m.l = m.h.createTexture());
                      if (
                        "undefined" !== typeof HTMLVideoElement &&
                        t instanceof HTMLVideoElement
                      ) {
                        var z = t.videoWidth;
                        x = t.videoHeight;
                      } else
                        "undefined" !== typeof HTMLImageElement &&
                        t instanceof HTMLImageElement
                          ? ((z = t.naturalWidth), (x = t.naturalHeight))
                          : ((z = t.width), (x = t.height));
                      x = { glName: m.l, width: z, height: x };
                      z = m.g;
                      z.canvas.width = x.width;
                      z.canvas.height = x.height;
                      z.activeTexture(z.TEXTURE0);
                      m.h.bindTexture2d(m.l);
                      z.texImage2D(
                        z.TEXTURE_2D,
                        0,
                        z.RGBA,
                        z.RGBA,
                        z.UNSIGNED_BYTE,
                        t
                      );
                      m.h.bindTexture2d(0);
                      m = x;
                      break a;
                    case "detections":
                      x = m.m[n.stream];
                      x || ((x = new Rb(m.h)), (m.m[n.stream] = x));
                      m = x;
                      m.data || (m.data = new m.g.DetectionListData());
                      m.data.reset(t.length);
                      for (x = 0; x < t.length; ++x) {
                        z = t[x];
                        var E = m.data,
                          F = E.setBoundingBox,
                          I = x;
                        var H = z.T;
                        var p = new Fb();
                        X(p, 1, H.Z);
                        X(p, 2, H.$);
                        X(p, 3, H.height);
                        X(p, 4, H.width);
                        X(p, 5, H.rotation);
                        X(p, 6, H.X);
                        var A = (H = new Ya());
                        U(A, 1, W(p, 1));
                        U(A, 2, W(p, 2));
                        U(A, 3, W(p, 3));
                        U(A, 4, W(p, 4));
                        U(A, 5, W(p, 5));
                        var C = W(p, 6);
                        if (null != C && null != C) {
                          Ra(A.g, 48);
                          var q = A.g,
                            B = C;
                          C = 0 > B;
                          B = Math.abs(B);
                          var D = B >>> 0;
                          B = Math.floor((B - D) / 4294967296);
                          B >>>= 0;
                          C &&
                            ((B = ~B >>> 0),
                            (D = (~D >>> 0) + 1),
                            4294967295 < D &&
                              ((D = 0), B++, 4294967295 < B && (B = 0)));
                          Q = D;
                          R = B;
                          C = Q;
                          for (D = R; 0 < D || 127 < C; )
                            q.push((C & 127) | 128),
                              (C = ((C >>> 7) | (D << 25)) >>> 0),
                              (D >>>= 7);
                          q.push(C);
                        }
                        rb(p, A);
                        H = $a(H);
                        F.call(E, I, H);
                        if (z.O)
                          for (E = 0; E < z.O.length; ++E)
                            (p = z.O[E]),
                              (A = p.visibility ? !0 : !1),
                              (F = m.data),
                              (I = F.addNormalizedLandmark),
                              (H = x),
                              (p = Object.assign(Object.assign({}, p), {
                                visibility: A ? p.visibility : 0,
                              })),
                              (A = new Ab()),
                              X(A, 1, p.x),
                              X(A, 2, p.y),
                              X(A, 3, p.z),
                              p.visibility && X(A, 4, p.visibility),
                              (q = p = new Ya()),
                              U(q, 1, W(A, 1)),
                              U(q, 2, W(A, 2)),
                              U(q, 3, W(A, 3)),
                              U(q, 4, W(A, 4)),
                              U(q, 5, W(A, 5)),
                              rb(A, q),
                              (p = $a(p)),
                              I.call(F, H, p);
                        if (z.M)
                          for (E = 0; E < z.M.length; ++E) {
                            F = m.data;
                            I = F.addClassification;
                            H = x;
                            p = z.M[E];
                            A = new wb();
                            X(A, 2, p.Y);
                            p.index && X(A, 1, p.index);
                            p.label && X(A, 3, p.label);
                            p.displayName && X(A, 4, p.displayName);
                            q = p = new Ya();
                            D = W(A, 1);
                            if (null != D && null != D)
                              if ((Ra(q.g, 8), (C = q.g), 0 <= D)) Ra(C, D);
                              else {
                                for (B = 0; 9 > B; B++)
                                  C.push((D & 127) | 128), (D >>= 7);
                                C.push(1);
                              }
                            U(q, 2, W(A, 2));
                            C = W(A, 3);
                            null != C &&
                              ((C = Ca(C)),
                              Ra(q.g, 26),
                              Ra(q.g, C.length),
                              Za(q, q.g.end()),
                              Za(q, C));
                            C = W(A, 4);
                            null != C &&
                              ((C = Ca(C)),
                              Ra(q.g, 34),
                              Ra(q.g, C.length),
                              Za(q, q.g.end()),
                              Za(q, C));
                            rb(A, q);
                            p = $a(p);
                            I.call(F, H, p);
                          }
                      }
                      m = m.data;
                      break a;
                    default:
                      m = {};
                  }
                }
                u = m;
                w = n.stream;
                switch (n.type) {
                  case "video":
                    f.pushTexture2d(
                      Object.assign(Object.assign({}, u), {
                        stream: w,
                        timestamp: g,
                      })
                    );
                    break;
                  case "detections":
                    r = u;
                    r.stream = w;
                    r.timestamp = g;
                    f.pushDetectionList(r);
                    break;
                  default:
                    throw Error("Unknown input config type: '" + n.type + "'");
                }
              }
            e.i.send(f);
            return N(y, e.C, 4);
          case 4:
            f.delete(), (y.g = 0);
        }
      });
    });
  };
  function dc(a, b, c) {
    return Z(a, function e() {
      var g,
        f,
        h,
        k,
        l,
        n,
        u = this,
        w,
        r,
        y,
        m,
        t,
        x,
        z,
        E;
      return O(e, function (F) {
        switch (F.g) {
          case 1:
            if (!c) return F.return(b);
            g = {};
            f = 0;
            h = K(Object.keys(c));
            for (k = h.next(); !k.done; k = h.next())
              (l = k.value),
                (n = c[l]),
                "string" !== typeof n &&
                  "texture" === n.type &&
                  void 0 !== b[n.stream] &&
                  ++f;
            1 < f && (u.G = !1);
            w = K(Object.keys(c));
            k = w.next();
          case 2:
            if (k.done) {
              F.g = 4;
              break;
            }
            r = k.value;
            y = c[r];
            if ("string" === typeof y)
              return (z = g), (E = r), N(F, ec(u, r, b[y]), 14);
            m = b[y.stream];
            if ("detection_list" === y.type) {
              if (m) {
                var I = m.getRectList();
                for (
                  var H = m.getLandmarksList(),
                    p = m.getClassificationsList(),
                    A = [],
                    C = 0;
                  C < I.size();
                  ++C
                ) {
                  var q = I.get(C);
                  a: {
                    var B = new Fb();
                    for (q = new Sa(q); S(q); )
                      switch (q.i) {
                        case 13:
                          var D = T(q);
                          X(B, 1, D);
                          break;
                        case 21:
                          D = T(q);
                          X(B, 2, D);
                          break;
                        case 29:
                          D = T(q);
                          X(B, 3, D);
                          break;
                        case 37:
                          D = T(q);
                          X(B, 4, D);
                          break;
                        case 45:
                          D = T(q);
                          X(B, 5, D);
                          break;
                        case 48:
                          D = Oa(q.g);
                          X(B, 6, D);
                          break;
                        default:
                          if (!sb(B, q)) break a;
                      }
                  }
                  B = {
                    Z: Y(B, 1),
                    $: Y(B, 2),
                    height: Y(B, 3),
                    width: Y(B, 4),
                    rotation: Y(B, 5, 0),
                    X: lb(B, 6, 0),
                  };
                  q = nb(Eb(H.get(C)), Ab).map(Nb);
                  var la = p.get(C);
                  a: for (D = new yb(), la = new Sa(la); S(la); )
                    switch (la.i) {
                      case 10:
                        D.addClassification(Ua(la, new wb(), xb));
                        break;
                      default:
                        if (!sb(D, la)) break a;
                    }
                  B = { T: B, O: q, M: Mb(D) };
                  A.push(B);
                }
                I = A;
              } else I = [];
              g[r] = I;
              F.g = 7;
              break;
            }
            if ("proto_list" === y.type) {
              if (m) {
                I = Array(m.size());
                for (H = 0; H < m.size(); H++) I[H] = m.get(H);
                m.delete();
              } else I = [];
              g[r] = I;
              F.g = 7;
              break;
            }
            if (void 0 === m) {
              F.g = 3;
              break;
            }
            if ("float_list" === y.type) {
              g[r] = m;
              F.g = 7;
              break;
            }
            if ("proto" === y.type) {
              g[r] = m;
              F.g = 7;
              break;
            }
            if ("texture" !== y.type)
              throw Error("Unknown output config type: '" + y.type + "'");
            t = u.s[r];
            t || ((t = new Ob(u.h, u.D)), (u.s[r] = t));
            return N(F, Pb(t, m, u.G), 13);
          case 13:
            (x = F.h), (g[r] = x);
          case 7:
            y.transform && g[r] && (g[r] = y.transform(g[r]));
            F.g = 3;
            break;
          case 14:
            z[E] = F.h;
          case 3:
            k = w.next();
            F.g = 2;
            break;
          case 4:
            return F.return(g);
        }
      });
    });
  }
  function ec(a, b, c) {
    return Z(a, function e() {
      var g = this,
        f;
      return O(e, function (h) {
        return "number" === typeof c ||
          c instanceof Uint8Array ||
          c instanceof g.h.Uint8BlobList
          ? h.return(c)
          : c instanceof g.h.Texture2dDataOut
          ? ((f = g.s[b]),
            f || ((f = new Ob(g.h, g.D)), (g.s[b] = f)),
            h.return(Pb(f, c, g.G)))
          : h.return(void 0);
      });
    });
  }
  function bc(a, b) {
    for (
      var c = b.name || "$",
        d = [].concat(L(b.wants)),
        e = new a.h.StringList(),
        g = K(b.wants),
        f = g.next();
      !f.done;
      f = g.next()
    )
      e.push_back(f.value);
    g = a.h.PacketListener.implement({
      onResults: function (h) {
        for (var k = {}, l = 0; l < b.wants.length; ++l) k[d[l]] = h.get(l);
        var n = a.listeners[c];
        n &&
          (a.C = dc(a, k, b.outs).then(function (u) {
            u = n(u);
            for (var w = 0; w < b.wants.length; ++w) {
              var r = k[d[w]];
              "object" === typeof r &&
                r.hasOwnProperty &&
                r.hasOwnProperty("delete") &&
                r.delete();
            }
            u && (a.C = u);
          }));
      },
    });
    a.i.attachMultiListener(e, g);
    e.delete();
  }
  v.onResults = function (a, b) {
    this.listeners[b || "$"] = a;
  };
  P("Solution", Xb);
  P("OptionType", {
    BOOL: 0,
    NUMBER: 1,
    aa: 2,
    0: "BOOL",
    1: "NUMBER",
    2: "STRING",
  });
  function fc(a) {
    a = Kb(a);
    var b = a.getMesh();
    if (!b) return a;
    var c = new Float32Array(b.getVertexBufferList());
    b.getVertexBufferList = function () {
      return c;
    };
    var d = new Uint32Array(b.getIndexBufferList());
    b.getIndexBufferList = function () {
      return d;
    };
    return a;
  }
  var gc = {
    files: [
      { url: "face_mesh_solution_packed_assets_loader.js" },
      { simd: !0, url: "face_mesh_solution_simd_wasm_bin.js" },
      { simd: !1, url: "face_mesh_solution_wasm_bin.js" },
    ],
    graph: { url: "face_mesh.binarypb" },
    listeners: [
      {
        wants: [
          "multi_face_geometry",
          "image_transformed",
          "multi_face_landmarks",
        ],
        outs: {
          image: "image_transformed",
          multiFaceGeometry: {
            type: "proto_list",
            stream: "multi_face_geometry",
            transform: function (a) {
              return a.map(fc);
            },
          },
          multiFaceLandmarks: {
            type: "proto_list",
            stream: "multi_face_landmarks",
            transform: function (a) {
              return a.map(function (b) {
                return nb(Eb(b), Ab).map(Nb);
              });
            },
          },
        },
      },
    ],
    inputs: { image: { type: "video", stream: "input_frames_gpu" } },
    options: {
      useCpuInference: {
        type: 0,
        graphOptionXref: {
          calculatorType: "InferenceCalculator",
          fieldName: "use_cpu_inference",
        },
        default:
          "iPad Simulator;iPhone Simulator;iPod Simulator;iPad;iPhone;iPod"
            .split(";")
            .includes(navigator.platform) ||
          (navigator.userAgent.includes("Mac") && "ontouchend" in document),
      },
      enableFaceGeometry: {
        type: 0,
        graphOptionXref: {
          calculatorName: "EnableFaceGeometryConstant",
          calculatorType: "ConstantSidePacketCalculator",
          fieldName: "bool_value",
        },
      },
      selfieMode: {
        type: 0,
        graphOptionXref: {
          calculatorType: "GlScalerCalculator",
          calculatorIndex: 1,
          fieldName: "flip_horizontal",
        },
      },
      maxNumFaces: {
        type: 1,
        graphOptionXref: {
          calculatorType: "ConstantSidePacketCalculator",
          calculatorName: "ConstantSidePacketCalculatorNumFaces",
          fieldName: "int_value",
        },
      },
      refineLandmarks: {
        type: 0,
        graphOptionXref: {
          calculatorType: "ConstantSidePacketCalculator",
          calculatorName: "ConstantSidePacketCalculatorRefineLandmarks",
          fieldName: "bool_value",
        },
      },
      minDetectionConfidence: {
        type: 1,
        graphOptionXref: {
          calculatorType: "TensorsToDetectionsCalculator",
          calculatorName:
            "facelandmarkfrontgpu__facedetectionshortrangegpu__facedetectionshortrangecommon__TensorsToDetectionsCalculator",
          fieldName: "min_score_thresh",
        },
      },
      minTrackingConfidence: {
        type: 1,
        graphOptionXref: {
          calculatorType: "ThresholdingCalculator",
          calculatorName:
            "facelandmarkfrontgpu__facelandmarkgpu__ThresholdingCalculator",
          fieldName: "threshold",
        },
      },
      cameraNear: {
        type: 1,
        graphOptionXref: {
          calculatorType: "FaceGeometryEnvGeneratorCalculator",
          fieldName: "near",
        },
      },
      cameraFar: {
        type: 1,
        graphOptionXref: {
          calculatorType: "FaceGeometryEnvGeneratorCalculator",
          fieldName: "far",
        },
      },
      cameraVerticalFovDegrees: {
        type: 1,
        graphOptionXref: {
          calculatorType: "FaceGeometryEnvGeneratorCalculator",
          fieldName: "vertical_fov_degrees",
        },
      },
    },
  };
  var hc = [
      [61, 146],
      [146, 91],
      [91, 181],
      [181, 84],
      [84, 17],
      [17, 314],
      [314, 405],
      [405, 321],
      [321, 375],
      [375, 291],
      [61, 185],
      [185, 40],
      [40, 39],
      [39, 37],
      [37, 0],
      [0, 267],
      [267, 269],
      [269, 270],
      [270, 409],
      [409, 291],
      [78, 95],
      [95, 88],
      [88, 178],
      [178, 87],
      [87, 14],
      [14, 317],
      [317, 402],
      [402, 318],
      [318, 324],
      [324, 308],
      [78, 191],
      [191, 80],
      [80, 81],
      [81, 82],
      [82, 13],
      [13, 312],
      [312, 311],
      [311, 310],
      [310, 415],
      [415, 308],
    ],
    ic = [
      [263, 249],
      [249, 390],
      [390, 373],
      [373, 374],
      [374, 380],
      [380, 381],
      [381, 382],
      [382, 362],
      [263, 466],
      [466, 388],
      [388, 387],
      [387, 386],
      [386, 385],
      [385, 384],
      [384, 398],
      [398, 362],
    ],
    jc = [
      [276, 283],
      [283, 282],
      [282, 295],
      [295, 285],
      [300, 293],
      [293, 334],
      [334, 296],
      [296, 336],
    ],
    kc = [
      [33, 7],
      [7, 163],
      [163, 144],
      [144, 145],
      [145, 153],
      [153, 154],
      [154, 155],
      [155, 133],
      [33, 246],
      [246, 161],
      [161, 160],
      [160, 159],
      [159, 158],
      [158, 157],
      [157, 173],
      [173, 133],
    ],
    lc = [
      [46, 53],
      [53, 52],
      [52, 65],
      [65, 55],
      [70, 63],
      [63, 105],
      [105, 66],
      [66, 107],
    ],
    mc = [
      [10, 338],
      [338, 297],
      [297, 332],
      [332, 284],
      [284, 251],
      [251, 389],
      [389, 356],
      [356, 454],
      [454, 323],
      [323, 361],
      [361, 288],
      [288, 397],
      [397, 365],
      [365, 379],
      [379, 378],
      [378, 400],
      [400, 377],
      [377, 152],
      [152, 148],
      [148, 176],
      [176, 149],
      [149, 150],
      [150, 136],
      [136, 172],
      [172, 58],
      [58, 132],
      [132, 93],
      [93, 234],
      [234, 127],
      [127, 162],
      [162, 21],
      [21, 54],
      [54, 103],
      [103, 67],
      [67, 109],
      [109, 10],
    ],
    nc = [].concat(L(hc), L(ic), L(jc), L(kc), L(lc), L(mc));
  function oc(a) {
    a = a || {};
    a = Object.assign(Object.assign({}, gc), a);
    this.g = new Xb(a);
  }
  v = oc.prototype;
  v.close = function () {
    this.g.close();
    return Promise.resolve();
  };
  v.onResults = function (a) {
    this.g.onResults(a);
  };
  v.initialize = function () {
    return Z(this, function b() {
      var c = this;
      return O(b, function (d) {
        return N(d, c.g.initialize(), 0);
      });
    });
  };
  v.reset = function () {
    this.g.reset();
  };
  v.send = function (a) {
    return Z(this, function c() {
      var d = this;
      return O(c, function (e) {
        return N(e, d.g.send(a), 0);
      });
    });
  };
  v.setOptions = function (a) {
    this.g.setOptions(a);
  };
  P("FACE_GEOMETRY", {
    Layout: {
      COLUMN_MAJOR: 0,
      ROW_MAJOR: 1,
      0: "COLUMN_MAJOR",
      1: "ROW_MAJOR",
    },
    PrimitiveType: { TRIANGLE: 0, 0: "TRIANGLE" },
    VertexType: { VERTEX_PT: 0, 0: "VERTEX_PT" },
    DEFAULT_CAMERA_PARAMS: { verticalFovDegrees: 63, near: 1, far: 1e4 },
  });
  P("FaceMesh", oc);
  P("FACEMESH_LIPS", hc);
  P("FACEMESH_LEFT_EYE", ic);
  P("FACEMESH_LEFT_EYEBROW", jc);
  P("FACEMESH_LEFT_IRIS", [
    [474, 475],
    [475, 476],
    [476, 477],
    [477, 474],
  ]);
  P("FACEMESH_RIGHT_EYE", kc);
  P("FACEMESH_RIGHT_EYEBROW", lc);
  P("FACEMESH_RIGHT_IRIS", [
    [469, 470],
    [470, 471],
    [471, 472],
    [472, 469],
  ]);
  P("FACEMESH_FACE_OVAL", mc);
  P("FACEMESH_CONTOURS", nc);
  P("matrixDataToMatrix", function (a) {
    for (
      var b = a.getCols(),
        c = a.getRows(),
        d = a.getPackedDataList(),
        e = [],
        g = 0;
      g < c;
      g++
    )
      e.push(Array(b));
    for (g = 0; g < c; g++)
      for (var f = 0; f < b; f++) {
        var h = 1 === a.getLayout() ? g * b + f : f * c + g;
        e[g][f] = d[h];
      }
    return e;
  });
  P("VERSION", "0.4.1633559619");
}.call(this));
