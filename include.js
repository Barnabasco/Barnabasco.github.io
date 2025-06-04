// Load Header and Footer dynamically
document.addEventListener("DOMContentLoaded", function () {
    // Include Header
    fetch("header.html")
        .then(response => response.text())
        .then(data => {
            document.getElementById("header").innerHTML = data;
        });

    // Include Footer
    fetch("footer.html")
        .then(response => response.text())
        .then(data => {
            document.getElementById("footer").innerHTML = data;

            // Add Event Listener for Contact Form
            const contactForm = document.getElementById("contact-form");
            if (contactForm) {
                contactForm.addEventListener("submit", function (e) {
                    e.preventDefault();
                    alert("Thank you! Your message has been sent.");
                });
            }
        });
});
