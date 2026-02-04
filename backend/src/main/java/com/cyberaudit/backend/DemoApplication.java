package com.cyberaudit.backend;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

@SpringBootApplication
public class DemoApplication {

	public static void main(String[] args) {
		// This starts the Spring Boot application
		SpringApplication.run(DemoApplication.class, args);

		// Once started, you can access:
		// - H2 Console: http://localhost:8080/h2-console
		// - Health Check: http://localhost:8080/actuator/health
	}

}
