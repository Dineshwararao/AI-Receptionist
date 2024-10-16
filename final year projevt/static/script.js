document.getElementById("mainCircle").addEventListener("click", function() {
  var subItems = document.querySelectorAll(".sub-menu-item");
  subItems.forEach(function(item) {
      item.style.opacity = 1;
  });

  // Speech output
  const speech = new SpeechSynthesisUtterance("Welcome to the AI Receptionist.");
  speech.volume = 1;
  speech.rate = 1;
  speech.pitch = 1;
  window.speechSynthesis.speak(speech);
});

// Some random colors
const colors = ["#3CC157", "#2AA7FF", "#1B1B1B", "#FCBC0F", "#F85F36"];

const numBalls = 25;
const balls = [];

for (let i = 0; i < numBalls; i++) {
  let ball = document.createElement("div");
  ball.classList.add("ball");
  ball.style.background = colors[Math.floor(Math.random() * colors.length)];
  ball.style.left = `${Math.floor(Math.random() * 100)}vw`;
  ball.style.top = `${Math.floor(Math.random() * 100)}vh`;
  ball.style.transform = `scale(${Math.random()})`;
  ball.style.width = `8px`;
  ball.style.height = '8px';

  balls.push(ball);
  document.body.append(ball);
}

// Keyframes
balls.forEach((el, i, ra) => {
  let to = {
      x: Math.random() * (i % 2 === 0 ? -11 : 11),
      y: Math.random() * 12
  };

  let anim = el.animate(
      [
          { transform: "translate(0, 0)" },
          { transform: `translate(${to.x}rem, ${to.y}rem)` }
      ],
      {
          duration: (Math.random() + 2) * 5000, // random duration
          direction: "alternate",
          fill: "both",
          iterations: Infinity,
          easing: "ease-in-out"
      }
  );
});
