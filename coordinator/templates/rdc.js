// Generated by CoffeeScript 1.10.0
(function() {
  this.tsss = function() {
    return $('#p1').text('???');
  };

  $(function() {
    var h, ws;
    h = window.location.host;
    ws = new WebSocket("ws://" + h + "/ws_node_status");
    return ws.onmessage = function(e) {
      var msg;
      msg = JSON.parse(e.data);
      $('#ws').text(msg.localhost);
      console.log(msg);
      return ws.send('alive');
    };
  });

}).call(this);

//# sourceMappingURL=rdc.js.map