package com.cyberaudit.backend.controller;

import org.springframework.web.bind.annotation.*;

// This is a simple REST controller.
// It acts as a "hello world" endpoint to verify your backend runs properly.
// Later, this will handle requests from the React frontend to start vulnerability scans.

@RestController
@RequestMapping("/api")
public class HelloController {

    // Basic GET endpoint for testing the connection.
    // Accessing http://localhost:8080/api/hello should return a plain text message.
    @GetMapping("/hello")
    public String hello() {
        return "Backend running successfully!";
    }
}
