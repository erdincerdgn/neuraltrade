// "use client";

// import Head from "next/head";
// import * as Sentry from "@sentry/nextjs";
// import { useState, useEffect } from "react";

// class SentryExampleFrontendError extends Error {
//   constructor(message: string | undefined) {
//     super(message);
//     this.name = "SentryExampleFrontendError";
//   }
// }

// export default function Page() {
//   const [hasSentError, setHasSentError] = useState(false);
//   const [isConnected, setIsConnected] = useState(true);
  
//   useEffect(() => {
//     async function checkConnectivity() {
//       const result = await Sentry.diagnoseSdkConnectivity();
//       setIsConnected(result !== 'sentry-unreachable');
//     }
//     checkConnectivity();
//   }, []);

//   return (
//     <div>Will be continued...</div>
//   );
// }
