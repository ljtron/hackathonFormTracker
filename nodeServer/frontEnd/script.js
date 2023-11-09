// let video = document.querySelector("#webcam");

// if (navigator.mediaDevices.getUserMedia){
//      // passing arguemnt as an object
//      navigator.mediaDevices.getUserMedia({video : true})
//      .then (function (stream) {
//           // send data first
//           video.srcObject = stream;
//           console.log(stream);
//      })
//      // if sum go wrong
//      .catch (function(error) {
//           console.log("Something is up");
//      })
// }

// else{
//      console.log("getUserMedie does not support");
// }

// const socket = new WebSocket("ws://localhost:3000");

// socket.emit("message", { name: "John" });
// socket.addEventListener("message", function (event) {
//      const data = event.data;
//      console.log("Data received from backend:", data);
// });

// let testBTN = document.querySelector("#testBTN");
// testBTN.addEventListener("click", ()=>{
//     console.log("clicked")
// });

